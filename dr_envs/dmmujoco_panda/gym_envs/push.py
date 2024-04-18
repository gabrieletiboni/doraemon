from typing import Any, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy
import csv
import pdb
import json

import numpy as np
import gym
from scipy.spatial.transform import Rotation

import dr_envs
from dr_envs.dmmujoco_panda.core.panda_gym import PandaGymEnvironment
from dr_envs.dmmujoco_panda.core.controllers import Controller, \
                                                      FFJointPositionController, \
                                                      TorqueController
from dr_envs.dmmujoco_panda.core.interpolation import Repeater, AccelerationIntegrator
from dr_envs.dmmujoco_panda.core.utils import register_panda_env, distance_penalty

class PandaPushEnv(PandaGymEnvironment):
    """
    :description: The simple environment where the Panda robot is manipulating
        a box.
    """
    def __init__(self,
                 model_file,
                 action_interpolator: Repeater,
                 action_interpolator_kwargs: dict,
                 controller: Controller,
                 controller_kwargs: dict,
                 qacc_factor: float,
                 control_penalty_coeff: float,  # 1.
                 task_reward: str,  # "guide"
                 search_space_id: int,  # 0
                 norm_reward=False,
                 command_type="acc",
                 contact_penalties=False,
                 goal_low=np.array([.7, -.3]),
                 goal_high=np.array([1.2, .3]),
                 init_box_low=np.array([0.51, 0]),
                 init_box_high=np.array([0.51, 0.]),
                 init_box_jitter=0.0,
                 push_prec_alpha=1e-3,
                 init_jpos_jitter=0.2,
                 init_jvel_jitter=0.0,
                 box_height_jitter=0.0,
                 box_noise_stdev=0.0,
                 jvel_noise_stdev=0.0,
                 rotation_in_obs="none",
                 randomized_dynamics='mf',
                 absolute_acc_pen=False,
                 model_kwargs={}):
        """
            Currently ignored parameters:
                - command_type
        """
        self.search_space_id = int(search_space_id)
        self.absolute_acc_pen = absolute_acc_pen

        self.goal_low = goal_low
        self.goal_high = goal_high
        self.init_box_low = init_box_low
        self.init_box_high = init_box_high
        self.init_box_jitter = init_box_jitter
        # Optionally add a small jitter to the box height to regularize pushing, i.e
        # avoid exploiting the edge for pushing
        self.init_box_size = model_kwargs['box_size'] if 'box_size' in model_kwargs else None
        self.box_height_jitter = box_height_jitter
        self.box_noise_stdev = box_noise_stdev
        self.jvel_noise_stdev = jvel_noise_stdev
        
        # Optionally add box orientation to the state space
        if rotation_in_obs == "none":
            rot_dims = 0
        elif rotation_in_obs == "rotz":
            rot_dims = 1
        elif rotation_in_obs == "sincosz":
            rot_dims = 2
        else:
            raise ValueError("Invalid rotation_in_obs")
        self.rotation_in_obs = rotation_in_obs

        # Parent calls .reset() at the end of __init__()
        PandaGymEnvironment.__init__(self,
                                     model_file=model_file,
                                     model_kwargs=model_kwargs,
                                     action_interpolator=action_interpolator,
                                     action_interpolator_kwargs=action_interpolator_kwargs,
                                     qacc_factor=qacc_factor,
                                     controller=controller,
                                     controller_kwargs=controller_kwargs,
                                     control_penalty_coeff=control_penalty_coeff,
                                     init_jpos_jitter=init_jpos_jitter,
                                     init_jvel_jitter=init_jvel_jitter)



        max_obs = np.array([np.inf]*(7 + 7 + 2 + 2 + rot_dims))  # 7 jpos + 7jvel + 2 box pos + 2 goal pos + box_rot_dims
        self.observation_space = gym.spaces.Box(-max_obs, max_obs)
        self.push_prec_alpha = push_prec_alpha

        self.last_dist_from_target = 0  # delta_distance appended to info dict
        self.task_reward = task_reward

        self.contact_penalties_enabled = contact_penalties  # penalize contact-pairs proportional to penetration distance
        self.contact_penalties = [("box", "table", 1e2),
                                # ("panda_finger1", "box", 1e2),
                                # ("panda_finger2", "box", 1e2),
                                ("panda_finger1", "table", 3e7),
                                ("panda_finger2", "table", 3e7)]
        self.contact_penalties = [(self.model.geom(c[0]).id,
                                   self.model.geom(c[1]).id,
                                   c[0],
                                   c[1],
                                   c[2])
                                        for c in self.contact_penalties]


        """
            Randomized dynamics parameters
        """
        self._save_init_dynamics_values()

        self._current_task = None
        self.dyn_ind_to_name = {}
        self.dyn_type = randomized_dynamics
        self.set_random_dynamics_type(dyn_type=randomized_dynamics)        

        self.original_task = np.copy(self.get_task())
        self.nominal_values = np.copy(self.original_task)
        self.task_dim = self.get_task().shape[0]
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.norm_reward = norm_reward
        if self.norm_reward:
            self.reward_threshold = 150
        else:
            self.reward_threshold = 2500
        self.preferred_lr = 0.001

        self.wandb_extra_metrics = {'last_dist_from_target': 'box_goal_distance'}
        self.success_metric = 'negative_last_dist_from_target'


    def get_contacts(self):
        """
        :description: Find contacts included in ``self.contact_penalties`` and return
            their penetration distance times the penalization coefficient. This
            can be used for penalizing contacts between certain geoms in the
            reward function.
        :returns: A dictionary with the keys of the form ``c_name1-name2`` with
            values corresponding to their contact depths scaled by the penalty.
        """
        contact_values = [0] * len(self.contact_penalties)  # initialize contact penalities to zero
        contacts = self.data.contact  # list of all contact points (should be)
        
        for c in contacts:
            geom1 = c.geom1
            geom2 = c.geom2
            for i, (id1, id2, name1, name2, coeff) in \
                    enumerate(self.contact_penalties):
                if id1 == geom1 and id2 == geom2 or id1 == geom2 and id2 == geom1:  # check whether you want to penalize this pair
                    contact_values[i] += coeff * c.dist**2  # c.dist = penetration distance
                    break
        res = {f"c_{self.contact_penalties[i][2]}-{self.contact_penalties[i][3]}": cv \
                for i, cv in enumerate(contact_values)}
        return res

    def step(self, action):
        state, reward, done, info = super().step(action)

        box_vel = self.box_velp
        for dim, vel in zip("xyz", box_vel):
            info[f"puck_dpos_{dim}"] = vel*self.dt
        info[f"puck_dpos"] = np.sqrt(np.sum(box_vel**2)) * self.dt

        delta_dist_from_target = self.goal_dist - self.last_dist_from_target
        self.last_dist_from_target = self.goal_dist
        self.negative_last_dist_from_target = -self.last_dist_from_target
        info[f"goal_dist"] = self.goal_dist
        info[f"dgoal_dist"] = delta_dist_from_target

        # Calculate contact penalties
        if self.contact_penalties_enabled:
            contacts = self.get_contacts()
            info = dict(**info, **contacts)
            reward -= np.sum([v for _, v in contacts.items()])

        if self.verbose >= 1:
            print(f'---step reward: {reward}\n{json.dumps(info, sort_keys=True, indent=4)}')

        return state, reward, done, info

    def analyze_contacts(self):
        contacts = self.data.contact
        for c in contacts:
            if c.geom1 == c.geom2 == 0 and c.dist == 0:
                break
            # TODO: transform ids geom1 and geom2 to geom names
            print(f"{c.geom1} - {c.geom2}: {c.dist}")
        print("-"*80)

    def get_task_reward(self):
        guide_dist = np.sqrt(np.sum((self.box_pos - self.gripper_pos)**2))

        # Prevent both terms from reaching super high values if things go wrong
        # (it can make the training a bit unstable)
        guide_dist = min(guide_dist, 2)
        goal_dist = min(self.goal_dist, 2)

        if self.norm_reward:
            """
                f(d) = -x^2 -ln(x^2 + alpha)
                reward(d) = f(c*d)/a  # stretching and normalizing

                See more at: https://www.desmos.com/calculator/w12fnkzejn
            """
            c_goal, a_goal, alpha_goal =  3.12, 4.6, 0.01
            goal_term = distance_penalty(c_goal*goal_dist, alpha=alpha_goal)/a_goal

            c_guide, a_guide, alpha_guide = 7.75, 0.693, 0.5
            guide_term = distance_penalty(c_guide*guide_dist, alpha=alpha_guide)/a_guide
        else:
            goal_term = distance_penalty(goal_dist, alpha=self.push_prec_alpha)
            guide_term = distance_penalty(guide_dist, alpha=1e-1)

        if self.task_reward == "guide":
            return goal_term + 0.1 * guide_term
        elif self.task_reward == "lessguide":
            return goal_term + 0.01 * guide_term
        elif self.task_reward == "moreguide":
            return goal_term + 0.5 * guide_term
        elif self.task_reward == "reach":
            return guide_term
        elif self.task_reward == "target":
            return goal_term
        elif self.task_reward == None:
            return 0
        else:
            raise ValueError(f"Unknown reward type: {self.task_reward}")

    @property
    def goal_dist(self):
        goal_dist = np.sqrt(np.sum((self.box_pos[:2] - self.goal_pos[:2])**2))
        return goal_dist

    def reset(self):
        # Sample new dynamics and re_build model if necessary
        if self.dr_training:
            self.set_random_task()

        super().reset()
        self.set_random_goal()

        start_pos = np.random.uniform(self.init_box_low, self.init_box_high) + \
                    np.random.uniform([-self.init_box_jitter, -self.init_box_jitter], [self.init_box_jitter, self.init_box_jitter])
        self.box_pos = start_pos
        return self._get_obs()

    def set_random_goal(self):
        goal = np.random.uniform(self.goal_low, self.goal_high)
        self.goal_pos = goal[:2]


    def set_box_friction(self, value):
        """
        :description: Sets the friction between the box and the sliding surface
        :param value: New friction value. Can either be an array of 2 floats
            (to set the linear friction) or an array of 5 float (to set the
            torsional and rotational friction values as well)
        :raises ValueError: if the dim of ``value`` is other than 2 or 5
        """
        pair_fric = self.get_pair_friction("box", "table")
        if value.shape[0] == 1:
            pair_fric[:2] = np.repeat(value, 2)
        elif value.shape[0] == 2:
            # Only set linear friction
            pair_fric[:2] = value
        elif value.shape[0] == 3:
            # linear friction + torsional
            pair_fric[:3] = value
        elif value.shape[0] == 5:
            # Set all 5 friction components
            pair_fric[:] = value
        else:
            raise ValueError("Friction should be a vector of 2 or 5 elements.")

    def set_boxgripper_friction(self, value):
        """
        :description: Sets the friction between the box and panda gripper
        :param value: New friction value. Can either be an array of 2 floats
            (to set the linear friction) or an array of 5 float (to set the
            torsional and rotational friction values as well)
        :raises ValueError: if the dim of ``value`` is other than 2 or 5
        """
        pair_fric_finger1 = self.get_pair_friction("box", "panda_finger1")
        pair_fric_finger2 = self.get_pair_friction("box", "panda_finger2")
        if value.shape[0] == 2:
            # Only set linear friction
            pair_fric_finger1[:2] = value
            pair_fric_finger2[:2] = value
        elif value.shape[0] == 3:
            # linear friction + torsional
            pair_fric_finger1[:3] = value
            pair_fric_finger2[:3] = value
        elif value.shape[0] == 5:
            # Set all 5 friction components
            pair_fric_finger1[:] = value
            pair_fric_finger2[:] = value
        else:
            raise ValueError("Friction should be a vector of 2 or 5 elements.")

    @property
    def box_velp(self):
        """
        :return: the linear velocity of the box. The value is clamped to
            (-10, 10) to prevent training instabilities in case
            the simulation gets unstable.
        """
        box_vel = self.data.joint("box_joint").qvel[:3]
        box_vel = np.clip(box_vel, -10, 10)
        return np.array(box_vel)

    @box_velp.setter
    def box_velp(self, value):
        """
        :description: Sets the linear velocity of the object to the given value
        :param value: the new velocity of the box
        """
        value_dim = value.shape[0]
        self.data.joint("box_joint").qvel[:value_dim] = value[:]

    @property
    def box_pos(self):
        """
        :return: the position of the box. The value is clamped to (-10, 10)
            to prevent training instabilities in case the simulation gets
            unstable.
        """
        box_xyz = self.data.joint("box_joint").qpos[:3]
        box_xyz = np.clip(box_xyz, -2, 2)
        return np.array(box_xyz)

    @box_pos.setter
    def box_pos(self, value):
        """
        :description: Moves the box to a new position. If value is 3D, the XYZ
            coordinates of the box are changed; if it's 2D, only XY position
            is affected (and Z stays whatever it was).
        :param value: the new position of the box
        """
        value_dim = value.shape[0]
        self.data.joint("box_joint").qpos[:value_dim] = value[:]

    def set_box_pos(self, value):
        value_dim = value.shape[0]
        self.data.joint("box_joint").qpos[:value_dim] = value[:]

    @property
    def goal_pos(self):
        return np.array(self.data.geom("goal").xpos[:3])

    @goal_pos.setter
    def goal_pos(self, value):
        """Moves the goal pos to a new position"""
        value_dim = value.shape[0]
        geom_id = self.model.geom("goal").id
        self.model.geom_pos[geom_id][:value_dim] = value[:]

    @property
    def box_orientation(self):
        """
        :return: the orientation of the box as Euler angles, following
            the package-wide angle convention
        """
        box_joint_id = self.model.joint("box_joint").id
        
        # qpos is a quaternion, we need Euler angles
        quat = self.data.qpos[box_joint_id+3:box_joint_id+7]

        # note: scipy quaternion format is scalar-last, mujoco's is scalar-first
        scipy_quat = np.concatenate((quat[1:], [quat[0]]))
        transform = Rotation.from_quat(scipy_quat)

        # I suspect zyx is not the right convention.
        # Check out eulerseq attribute at https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler 
        # euler = transform.as_euler('zyx')  
        euler = transform.as_euler(dr_envs.dmmujoco_panda.EULER_ORDER)  # I did some tests and this is correct (xyz)

        return euler

    @box_orientation.setter
    def box_orientation(self, value):
        print('--- WARNING! scipy quaternions are in scalar-last format, while mujoco expects them with \
            scalar-first format. Make sure the value variable here is set correctly. See the \
            corresponding getter for an example.')
        raise NotImplementedError()

    def _get_obs(self):
        """Augment robot observations with box and goal"""
        if self.rotation_in_obs == "none":
            rot = []
        elif self.rotation_in_obs == "rotz":
            rot = [self.box_orientation[2]]
        elif self.rotation_in_obs == "sincosz":
            rot = [np.sin(self.box_orientation[2]), np.cos(self.box_orientation[2])]

        return np.concatenate([super()._get_obs(), self.box_pos[:2] + np.random.randn(2)*self.box_noise_stdev,
                               rot + np.random.randn(len(rot))*self.box_noise_stdev, self.goal_pos[:2]])


    def set_box_size(self, value: List[float]):
        """Sets the size of the box dimensions (meters)
            :param value: new size as HALF the edge lengths
                          Value can be [x,y,z] or [z].
        """
        assert isinstance(value, list) or isinstance(value, np.ndarray)

        if len(value) == 3:
            # Set all three sizes x,y,z
            pass
        elif len(value) == 1:
            # Set height only (z)
            value = [self.init_box_size[0], self.init_box_size[1], value[0]]

        self.model_kwargs["box_size"] = list(value)
        self._needs_rebuilding = True

    def set_box_com(self, value: List[float]):
        """Sets the center of mass of the hockey puck
            :param value: x,y,z list of new CoM offset w.r.t.
                          to geometric center
                          Value can also be [x,y] or [y].
        """
        assert isinstance(value, list) or isinstance(value, np.ndarray)

        if len(value) == 2:
            # Set com along z-axis to 0.0 (center)
            value = [value[0], value[1], 0.0]
        elif len(value) == 1:
            # Set comx = comz = 0.0, value is comy
            value = [0.0, value[0], 0.0]

        # self.model_kwargs["box_com"] = list(value)
        self.model_kwargs["box_com"] = " ".join([str(elem) for elem in value])
        self._needs_rebuilding = True

    def set_box_mass(self, new_mass):
        """
        :description: Sets the mass of the hockey puck
        :param mass: The new mass (float)
        """
        assert new_mass >= 0
        self.model_kwargs["box_mass"] = float(new_mass)
        self._needs_rebuilding = True

    def get_box_mass(self):
        return np.array(self.model.body("com").mass)[0]

    def get_task(self):
        return self._current_task

    def set_task(self, *task):
        if len(task) != len(self.dyn_ind_to_name.values()):
            raise ValueError(f"The task given is not compatible with the dyn type selected. dyn_type:{self.dyn_type} - task:{task}")
        
        task = np.array(task)
        self._current_task = np.array(task)

        if self.box_height_jitter > 0.0:
            # mujoco expects size as length from center (halved)
            jitter = np.random.uniform(-self.box_height_jitter, self.box_height_jitter) / 2
            self.set_box_size([self.init_box_size[2]+jitter])

        if self.dyn_type == 'm':
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()

        elif self.dyn_type == 'mf':
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:3])

        elif self.dyn_type == 'mft':
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:4])

        elif self.dyn_type == 'mfcom':
            self.set_box_com(task[3:5])
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:3])

        elif self.dyn_type == 'mfcomy':
            self.set_box_com(task[3:4])
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:3])

        elif self.dyn_type == 'com':
            self.set_box_com(task[:])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()

        elif self.dyn_type == 'comy':
            self.set_box_com(task[:])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()

        elif self.dyn_type == 'mftcom':
            self.set_box_com(task[4:6])
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:4])

        elif self.dyn_type == 'mfcomd':
            self.set_box_com(task[3:5])
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:3])
            self.set_joint_damping(task[5:12])

        elif self.dyn_type == 'd':
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_joint_damping(task[:7])

        elif self.dyn_type == 'd_fl':
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_joint_damping(task[:7])
            self.set_joint_frictionloss(task[7:14])

        elif self.dyn_type == 'mfcomy_d_fl':
            self.set_box_com(task[3:4])
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:3])
            self.set_joint_damping(task[4:11])
            self.set_joint_frictionloss(task[11:18])

        elif self.dyn_type == 'mf0comy_d_fl':
            self.set_box_com(task[2:3])
            self.set_box_mass(task[0])
            # Make sure you rebuild the model before changing other mj parameters, otherwise they'll get overridden
            if self._needs_rebuilding:
                self._rebuild_model()
            self.set_box_friction(task[1:2])
            self.set_joint_damping(task[3:10])
            self.set_joint_frictionloss(task[10:17])

        else:
            raise NotImplementedError(f"Current randomization type is not implemented (3): {self.dyn_type}")
        return

    def set_random_dynamics_type(self, dyn_type='mf'):
        """Selects which dynamics to be randomized
        with the corresponding name encoding
        """
        if dyn_type == 'm':  # mass
            self.dyn_ind_to_name = {0: 'mass'}
        elif dyn_type == 'mf':  # mass + friction
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony'}
        elif dyn_type == 'mft':  # mass + friction + torsional friction
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'frictiont'}
        elif dyn_type == 'mfcom':  # mass + friction + CoM
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'comx', 4: 'comy'}
        elif dyn_type == 'mfcomy':  # mass + friction + CoMy
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'comy'}
        elif dyn_type == 'com':  # CoM
            self.dyn_ind_to_name = {0: 'comx', 1: 'comy'}
        elif dyn_type == 'comy':  # CoM
            self.dyn_ind_to_name = {0: 'comy'}
        elif dyn_type == 'mftcom':  # mass + friction + torsional + CoM
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'frictiont', 4: 'comx', 5: 'comy'}
        elif dyn_type == 'mfcomd':  # + joint dampings
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'comx', 4: 'comy', 5: 'damping0', 6: 'damping1', 7: 'damping2', 8: 'damping3', 9: 'damping4', 10: 'damping5', 11: 'damping6'}
        elif dyn_type == 'd':  # joint dampings
            self.dyn_ind_to_name = {0: 'damping0', 1: 'damping1', 2: 'damping2', 3: 'damping3', 4: 'damping4', 5: 'damping5', 6: 'damping6'}
        elif dyn_type == 'd_fl':  # joint dampings and joint frictionloss
            self.dyn_ind_to_name = {0: 'damping0', 1: 'damping1', 2: 'damping2', 3: 'damping3', 4: 'damping4', 5: 'damping5', 6: 'damping6',
                                    7: 'frictionloss0', 8: 'frictionloss1', 9: 'frictionloss2', 10: 'frictionloss3', 11: 'frictionloss4', 12: 'frictionloss5', 13: 'frictionloss6'}
        elif dyn_type == 'mfcomy_d_fl':  # mass + friction x,y + comy + joint dampings and joint frictionloss
            self.dyn_ind_to_name = {0: 'mass', 1: 'frictionx', 2: 'frictiony', 3: 'comy', 
                                    4: 'damping0', 5: 'damping1', 6: 'damping2', 7: 'damping3', 8: 'damping4', 9: 'damping5', 10: 'damping6',
                                    11: 'frictionloss0', 12: 'frictionloss1', 13: 'frictionloss2', 14: 'frictionloss3', 15: 'frictionloss4', 16: 'frictionloss5', 17: 'frictionloss6'}
        elif dyn_type == 'mf0comy_d_fl':  # mass + friction for both axes + comy + joint dampings and joint frictionloss
            self.dyn_ind_to_name = {0: 'mass', 1: 'friction', 2: 'comy', 
                                    3: 'damping0', 4: 'damping1', 5: 'damping2', 6: 'damping3', 7: 'damping4', 8: 'damping5', 9: 'damping6',
                                    10: 'frictionloss0', 11: 'frictionloss1', 12: 'frictionloss2', 13: 'frictionloss3', 14: 'frictionloss4', 15: 'frictionloss5', 16: 'frictionloss6'}

        else:
            raise NotImplementedError(f"Randomization dyn_type not implemented: {dyn_type}")

        # Safety check that above dicts are set properly
        assert (len(self.dyn_ind_to_name.values())-1) in self.dyn_ind_to_name and len(set(self.dyn_ind_to_name.keys())) == len(self.dyn_ind_to_name.values())

        self.dyn_type = dyn_type

        self._current_task = np.array(self.get_default_task())

        self.original_task = np.copy(self.get_task())
        self.task_dim = self.get_task().shape[0]
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)
        return

    def get_search_bounds_mean(self, index=-1, name=None):
        """Get search bounds for the mean of the parameters optimized"""
        
        if self.search_space_id == 0:
            # Difficult, starts with high frictions
            search_bounds_mean = {
                   'mass': (0.2, 0.6),
                   'friction':  (0.025, .5),
                   'frictionx': (0.025, .3),
                   'frictiony': (0.025, .3),
                   'frictiont': (0.001, 0.5),
                   'solref0': (0.001, 0.02),
                   'solref1': (0.4, 1.),
                   'comx': (-0.05, 0.05),
                   'comy': (-0.05, 0.05),
                   'damping0': (0.5, 2.5),
                   'damping1': (0.5, 2.5),
                   'damping2': (0.5, 2.5),
                   'damping3': (0.5, 2.5),
                   'damping4': (0.5, 2.5),
                   'damping5': (0.5, 2.5),
                   'damping6': (0.5, 2.5),
                   'frictionloss0': (1., 3.),
                   'frictionloss1': (1., 3.),
                   'frictionloss2': (1., 3.),
                   'frictionloss3': (1., 3.),
                   'frictionloss4': (1., 3.),
                   'frictionloss5': (1., 3.),
                   'frictionloss6': (1., 3.),
            }
        elif self.search_space_id == 1:
            # Wider, includes low frictions
            search_bounds_mean = {
                   'mass': (0.2, 0.6),
                   'friction':  (0.025, .5),
                   'frictionx': (0.025, .3),
                   'frictiony': (0.025, .3),
                   'frictiont': (0.001, 0.5),
                   'solref0': (0.001, 0.02),
                   'solref1': (0.4, 1.),
                   'comx': (-0.05, 0.05),
                   'comy': (-0.05, 0.05),
                   'damping0': (0.025, 2.5),
                   'damping1': (0.025, 2.5),
                   'damping2': (0.025, 2.5),
                   'damping3': (0.025, 2.5),
                   'damping4': (0.025, 2.5),
                   'damping5': (0.025, 2.5),
                   'damping6': (0.025, 2.5),
                   'frictionloss0': (0.025, 3.),
                   'frictionloss1': (0.025, 3.),
                   'frictionloss2': (0.025, 3.),
                   'frictionloss3': (0.025, 3.),
                   'frictionloss4': (0.025, 3.),
                   'frictionloss5': (0.025, 3.),
                   'frictionloss6': (0.025, 3.),
            }
        elif self.search_space_id == 2:
            # FOR UDR: EASY FIXED FRICTIONS
            search_bounds_mean = {
                   'mass': (0.2, 0.6),
                   'friction':  (0.025, .5),
                   'frictionx': (0.025, .3),
                   'frictiony': (0.025, .3),
                   'frictiont': (0.001, 0.5),
                   'solref0': (0.001, 0.02),
                   'solref1': (0.4, 1.),
                   'comx': (-0.05, 0.05),
                   'comy': (-0.05, 0.05),
                   'damping0': (0.1, 0.1),
                   'damping1': (0.1, 0.1),
                   'damping2': (0.1, 0.1),
                   'damping3': (0.1, 0.1),
                   'damping4': (0.1, 0.1),
                   'damping5': (0.1, 0.1),
                   'damping6': (0.1, 0.1),
                   'frictionloss0': (0.1, 0.1),
                   'frictionloss1': (0.1, 0.1),
                   'frictionloss2': (0.1, 0.1),
                   'frictionloss3': (0.1, 0.1),
                   'frictionloss4': (0.1, 0.1),
                   'frictionloss5': (0.1, 0.1),
                   'frictionloss6': (0.1, 0.1),
            }
        elif self.search_space_id == 3:
            # FOR UDR: MEDIUM FIXED FRICTIONS
            search_bounds_mean = {
                   'mass': (0.2, 0.6),
                   'friction':  (0.025, .5),
                   'frictionx': (0.025, .3),
                   'frictiony': (0.025, .3),
                   'frictiont': (0.001, 0.5),
                   'solref0': (0.001, 0.02),
                   'solref1': (0.4, 1.),
                   'comx': (-0.05, 0.05),
                   'comy': (-0.05, 0.05),
                   'damping0': (0.6, 0.6),
                   'damping1': (0.6, 0.6),
                   'damping2': (0.6, 0.6),
                   'damping3': (0.6, 0.6),
                   'damping4': (0.6, 0.6),
                   'damping5': (0.6, 0.6),
                   'damping6': (0.6, 0.6),
                   'frictionloss0': (0.8, 0.8),
                   'frictionloss1': (0.8, 0.8),
                   'frictionloss2': (0.8, 0.8),
                   'frictionloss3': (0.8, 0.8),
                   'frictionloss4': (0.8, 0.8),
                   'frictionloss5': (0.8, 0.8),
                   'frictionloss6': (0.8, 0.8),
            }
        else:
            raise ValueError(f'Search space id  {self.search_space_id} is not valid.')
        if name is None:
            return search_bounds_mean[self.dyn_ind_to_name[index]]
        else:
            return search_bounds_mean[name]

    def get_task_lower_bound(self, index):
        """Returns lowest possible feasible value for each dynamics"""
        lowest_value = {
                    'mass': 0.02, # 20gr
                    'friction':  0.01,
                    'frictionx': 0.01,
                    'frictiony': 0.01,
                    'frictiont': 0.001,
                    'comx': -0.05,
                    'comy': -0.05,
                    'damping0': 0.,
                    'damping1': 0.,
                    'damping2': 0.,
                    'damping3': 0.,
                    'damping4': 0.,
                    'damping5': 0.,
                    'damping6': 0.,
                    'frictionloss0': 0.,
                    'frictionloss1': 0.,
                    'frictionloss2': 0.,
                    'frictionloss3': 0.,
                    'frictionloss4': 0.,
                    'frictionloss5': 0.,
                    'frictionloss6': 0.,
        }
        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task_upper_bound(self, index):
        """Returns highest possible feasible value for each dynamics"""
        highest_value = {
                    'mass': 2.0, #2kg
                    'friction':  2.,
                    'frictionx': 2.,
                    'frictiony': 2.,
                    'frictiont': 1,
                    'comx': 0.05,
                    'comy': 0.05,
                    'damping0': 1000,
                    'damping1': 1000,
                    'damping2': 1000,
                    'damping3': 1000,
                    'damping4': 1000,
                    'damping5': 1000,
                    'damping6': 1000,
                    'frictionloss0': 1000,
                    'frictionloss1': 1000,
                    'frictionloss2': 1000,
                    'frictionloss3': 1000,
                    'frictionloss4': 1000,
                    'frictionloss5': 1000,
                    'frictionloss6': 1000,

        }
        return highest_value[self.dyn_ind_to_name[index]]


    def get_default_task(self):
        default_values = {
            'mass': self.init_box_mass,
            'friction': self.init_box_table_friction[0],
            'frictionx': self.init_box_table_friction[0],
            'frictiony': self.init_box_table_friction[1],
            'frictiont': self.init_box_table_friction[2],
            'comx': self.init_box_com[0],
            'comy': self.init_box_com[1],
            'damping0': self.init_joint_damping[0],
            'damping1': self.init_joint_damping[1],
            'damping2': self.init_joint_damping[2],
            'damping3': self.init_joint_damping[3],
            'damping4': self.init_joint_damping[4],
            'damping5': self.init_joint_damping[5],
            'damping6': self.init_joint_damping[6],
            'frictionloss0': self.init_joint_frictionloss[0],
            'frictionloss1': self.init_joint_frictionloss[1],
            'frictionloss2': self.init_joint_frictionloss[2],
            'frictionloss3': self.init_joint_frictionloss[3],
            'frictionloss4': self.init_joint_frictionloss[4],
            'frictionloss5': self.init_joint_frictionloss[5],
            'frictionloss6': self.init_joint_frictionloss[6],
        }
        default_task = [default_values[dyn] for dyn in self.dyn_ind_to_name.values()]
        return default_task

    def _save_init_dynamics_values(self):
        """Saves initial dynamics parameter values"""
        self.init_joint_damping = self.get_joint_damping()
        self.init_joint_frictionloss = self.get_joint_frictionloss()
        self.init_box_table_friction = np.array(self.get_pair_friction("box", "table"))
        self.init_box_com = self._from_str_to_float(self.model_kwargs["box_com"]) if "box_com" in self.model_kwargs else np.array([0.0, 0.0, 0.0])
        self.init_box_mass = self.get_box_mass()


    def _from_str_to_float(self, string):
        values = []
        for item in string.split():
            values.append(float(item))
        return values



"""


    Gym-registered panda push environments


"""
panda_start_jpos = np.array([0, 0.15, 0, -2.60, 0, 1.20, 0])
fixed_push_goal_a = np.array([0.75, 0.0])
goal_ranges = {
                    'GoalA': (fixed_push_goal_a, fixed_push_goal_a),
                    'RandGoal0': (np.array([0.6, -0.2]), np.array([0.75, 0.2])),
                    'RandGoal1': (np.array([0.58, -0.12]), np.array([0.77, 0.12])),
                    'RandGoalDebug': (np.array([0.749, -0.001]), np.array([0.75, 0.001]))  # debug fixed goal
              }

randomized_dynamics = ['m', 'mf', 'mft', 'mfcom', 'mfcomy', 'com', 'comy', 'mftcom', 'mfcomd', 'd', 'd_fl', 'mfcomy_d_fl', 'mf0comy_d_fl']
norm_reward_bool=[True, False]
# task_rewards = ['target', 'guide']
init_jpos_jitters = [0.0, 0.02]
init_box_jitters = [0.0, 0.01]
box_height_jitters = [0.0, 0.005]
box_noise_stdevs = [0.0, 0.002]
jvel_noise_stdevs = [0.0, 0.0011]
clip_accelerations = [True, False]
# qacc_factors = [None, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
contact_penalties = [False, True]
# ctrl_pen_coeffs = [None, 0.1, 0.2, 0.5, 1.0, 2.0]

# Simple env for debugging
register_panda_env(
        id="DMPandaPush-FFPosCtrl-GoalA-v0",
        entry_point="%s:PandaPushEnv" % __name__,
        model_file="TableBoxScene.xml",
        controller=FFJointPositionController,
        controller_kwargs = {"clip_acceleration": False, "velocity_noise": True},
        action_interpolator=AccelerationIntegrator,
        action_interpolator_kwargs={"velocity_noise": True},
        model_kwargs = {"with_goal": True,
                        "init_joint_pos": panda_start_jpos,
                        "box_size": [0.05, 0.05, 0.04]},
        max_episode_steps=300,
        env_kwargs = {"command_type": "acc",
                      "contact_penalties": False,
                      "control_penalty_coeff": 1.,
                      "task_reward": "target",
                      "goal_low": fixed_push_goal_a,
                      "goal_high": fixed_push_goal_a,
                      "init_jpos_jitter": 0.,
                      "norm_reward": True,
                      "rotation_in_obs": "sincosz",
                      "box_height_jitter": 0.
            }
        )

for dyn_type in randomized_dynamics:
    for norm_reward in norm_reward_bool:
        # for task_reward in task_rewards:
        for init_jpos_jitter in init_jpos_jitters:
            for init_box_jitter in init_box_jitters:
                for box_height_jitter in box_height_jitters:
                    for goal_name, goal_range in goal_ranges.items():
                        for clip_acc in clip_accelerations:
                            # for qacc_factor in qacc_factors:
                                # for ctrl_pen_coeff in ctrl_pen_coeffs:
                                for box_noise_stdev in box_noise_stdevs:
                                    for jvel_noise_stdev in jvel_noise_stdevs:
                                        for contact_pen in contact_penalties:
                                            register_panda_env(
                                                    id=f"DMPandaPush-FFPosCtrl{'-ContPen' if contact_pen else ''}{'-ClipAcc' if clip_acc else ''}-{goal_name}-{dyn_type}{'-JVelNoise'+str(jvel_noise_stdev) if jvel_noise_stdev != 0. else ''}{'-BoxNoise'+str(box_noise_stdev) if box_noise_stdev != 0. else ''}{'-InitJpos'+str(init_jpos_jitter) if init_jpos_jitter != 0. else ''}{'-InitBox'+str(init_box_jitter) if init_box_jitter != 0. else ''}{'-BoxHeight'+str(box_height_jitter) if box_height_jitter != 0. else ''}{('-NormReward' if norm_reward else '')}-v0",
                                                    entry_point="%s:PandaPushEnv" % __name__,
                                                    model_file="TableBoxScene.xml",
                                                    controller=FFJointPositionController,
                                                    controller_kwargs={"clip_acceleration": clip_acc, "velocity_noise": True},
                                                    action_interpolator=AccelerationIntegrator,
                                                    action_interpolator_kwargs={"velocity_noise": True},
                                                    model_kwargs = {"actuator_type": "torque",
                                                                    "with_goal": True,
                                                                    "display_goal_range": True if (goal_range[1][0]-goal_range[0][0])/2 > 0. else False,  # display only if randomizing the goal
                                                                    "goal_range_center": (goal_range[0]+goal_range[1])/2,
                                                                    "goal_range_size": np.array([(goal_range[1][0]-goal_range[0][0])/2, (goal_range[1][1]-goal_range[0][1])/2]),
                                                                    "init_joint_pos": panda_start_jpos,
                                                                    "box_size": [0.05, 0.05, 0.04]},
                                                    max_episode_steps=300,
                                                    env_kwargs = {"command_type": "acc",
                                                                  # "qacc_factor": 0.3,
                                                                  "contact_penalties": contact_pen,
                                                                  # "control_penalty_coeff": 1.,
                                                                  # "task_reward": 'target',
                                                                  "norm_reward": norm_reward,
                                                                  "goal_low": goal_range[0],
                                                                  "goal_high": goal_range[1],
                                                                  "init_jpos_jitter": init_jpos_jitter,
                                                                  "init_box_jitter": init_box_jitter,
                                                                  "box_height_jitter": box_height_jitter,
                                                                  "box_noise_stdev": box_noise_stdev,
                                                                  "jvel_noise_stdev": jvel_noise_stdev,
                                                                  "rotation_in_obs": "sincosz",
                                                                  "randomized_dynamics": dyn_type,
                                                                  # "search_space_id": 0
                                                        }
                                                    )