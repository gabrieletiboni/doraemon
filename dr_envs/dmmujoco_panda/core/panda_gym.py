"""Test Franka Panda with deepmind Mujoco.

    Test of a Position Controller with feedforward to check whether
    it reproduces the real Panda controller in the lab.
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb
import os

import numpy as np
from scipy.optimize import minimize
try:
    import mujoco
except ImportError:
    print('Warning! Unable to import mujoco.')
    pass
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import gym

from dr_envs.dmmujoco_panda.core.template_renderer import TemplateRenderer
from dr_envs.dmmujoco_panda.core.controllers import Controller, \
                                                        FFJointPositionController, \
                                                        TorqueController
from dr_envs.dmmujoco_panda.core.interpolation import Repeater, AccelerationIntegrator
from dr_envs.dmmujoco_panda.core.utils import register_panda_env, \
                                                soft_tanh_limit, \
                                                square_penalty_limit
from dr_envs.random_env import RandomEnv

DEFAULT_SIZE = 1280



class PandaGymEnvironment(RandomEnv):
    def __init__(self,
                 model_file,
                 model_kwargs: dict,
                 action_interpolator: Repeater,
                 action_interpolator_kwargs: dict,
                 qacc_factor: float,
                 controller: Controller,
                 controller_kwargs: dict,
                 n_frames=1,
                 ctrl_frequency=50,
                 width: int = DEFAULT_SIZE,
                 height: int = DEFAULT_SIZE,
                 camera_id: Optional[int] = 1,
                 camera_name: Optional[str] = "side_camera",
                 default_camera_config: Optional[dict] = None,
                 init_jpos_jitter=0.0,
                 init_jvel_jitter=0.0,
                 control_penalty_coeff=1.):
        RandomEnv.__init__(self)

        self.width = width
        self.height = height
        self.model_file = model_file
        self.model_kwargs = model_kwargs
        self.ctrl_frequency = ctrl_frequency
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.default_camera_config = default_camera_config
        self.n_frames = n_frames
        self._initialize_simulation(init_values_to_default=True)
        self.qacc_factor = qacc_factor
        self._set_limits()

        assert init_jpos_jitter == 0.0 and init_jvel_jitter == 0.0, 'TODO: they are not yet implemented'
        
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.joint_qpos_shift = np.array([0, 0, 0, 0, 0, -np.pi/2, -np.pi/4])
        self.init_panda_qpos = np.array([0., 0.15, 0., -2.60, 0., 1.20, 0.]) - self.joint_qpos_shift
        self.init_panda_qvel = np.array([0., 0., 0., 0., 0., 0., 0.])

        self.action_interpolator = action_interpolator
        self.action_interpolator_kwargs = dict(action_interpolator_kwargs)
        self.interpolator = self._build_interpolator()
        self.controller = controller(self, **controller_kwargs)

        max_obs = np.array([np.inf]*14)
        max_action = np.ones(7)
        self.action_space = gym.spaces.Box(-max_action, max_action)
        self.observation_space = gym.spaces.Box(-max_obs, max_obs)

        self.control_penalty_coeff = control_penalty_coeff  # penalize pos, vel and acc when they are close to the limits

        self.verbose = 0
        self._needs_rebuilding = False
        self.reset()

    def set_verbosity(self, verbose):
        self.verbose = verbose

    def _initialize_simulation(self, init_values_to_default=False):
        ### Set initial values for dynamics that REQUIRE rebuilding the model
        if init_values_to_default:
            # This should not be done when DR is true, as it would overwrite the task that
            # DR put in self.model_kwargs. It's only meant to be done at the beginning to set
            # the default task.
            self.model_kwargs['box_mass'] = np.mean(list(self.get_search_bounds_mean(name='mass')))
            init_box_com = [
                            np.mean(list(self.get_search_bounds_mean(name='comx'))),
                            np.mean(list(self.get_search_bounds_mean(name='comy'))),
                            0.
                           ]
            self.model_kwargs["box_com"] = " ".join([str(elem) for elem in init_box_com])

        parsed_xml = self.parse_xml(self.model_file, **self.model_kwargs)
        
        self.model = mujoco.MjModel.from_xml_string(parsed_xml)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        self.mujoco_renderer = MujocoRenderer(
                                    self.model, self.data, self.default_camera_config
                               )

        self.arm_joint_index = [self.model.joint(f"joint{i+1}").qposadr[0] for i in range(7)]
        self.arm_act_index = [self.model.actuator(f"actuator{i+1}").id for i in range(7)]

        ### Set initial values for dynamics that DO NOT require rebuilding the model.
        # These values can be set even when DR is true, because they will be overwritten
        # after this function has finished.
        friction_init = np.mean(list(self.get_search_bounds_mean(name='friction')))
        self.set_box_friction(np.array([friction_init]))
        damping_init = np.mean(list(self.get_search_bounds_mean(name='damping0')))
        self.set_joint_damping(np.repeat(damping_init, 7))
        frictionloss_init = np.mean(list(self.get_search_bounds_mean(name='frictionloss0')))
        self.set_joint_frictionloss(np.repeat(frictionloss_init, 7))


    def _rebuild_model(self):
        self._initialize_simulation()
        self.interpolator = self._build_interpolator()
        self._needs_rebuilding = False
    
    def call_rebuild_model(self):
        self._rebuild_model()

    def parse_xml(self, template_file, **kwargs):
        """
        :description: Parse XML given Jinja template file,
            passing ``kwargs`` as template arguments.
        :param template_file: the name of the XML template file to be rendered
        :param **kwargs: keyword arguments which will be passed to the template
        :return: parsed xml
        """
        renderer = TemplateRenderer()
        xml_data = renderer.render_template(template_file, **kwargs)
        return xml_data

    def _build_interpolator(self):
        assert np.isclose((1/self.ctrl_frequency) % self.dt, 0), f'A control timestep of {1/self.ctrl_frequency}s cannot be split into a whole number of sim steps every {self.dt}s. Ratio={(1/self.ctrl_frequency) / self.dt}'
        self.num_ctrl_steps = int((1/self.ctrl_frequency)/self.dt) 
        return self.action_interpolator(num=self.num_ctrl_steps,
                                        dt=float(self.dt),
                                        **self.action_interpolator_kwargs)

    def step(self, action):
        """
            action: e.g. acceleration in [-1, 1] from tanh
        """
        action = np.array(action)  # may no longer be needed

        # Normalized acc [-1, 1] -> ctrl_format (e.g. (des_pos, des_vel, des_acc) in 20ms)
        action = self.preprocess_action(action)

        for _target in self.interpolator(action):
            # (des_pos, des_vel, des_acc) -> torques 
            control = self.controller.get_control(_target)
            self._step_mujoco_simulation(control, self.n_frames)

        state = self._get_obs()
        task_reward = self.get_task_reward()
        norm_acc = np.abs(self.joint_acc)/self.joint_qacc_max
        norm_vel = np.abs(self.joint_vel)/self.joint_qvel_max
        position_penalty = 0
        velocity_penalty = square_penalty_limit(self.joint_vel, self.joint_qvel_min, self.joint_qvel_max, square_coeff=5.).sum()
        if self.absolute_acc_pen:
            acceleration_penalty = soft_tanh_limit(self.joint_acc,
                -self.joint_qacc_max/self.qacc_factor, self.joint_qacc_max/self.qacc_factor, betas=(0.2, 0.2), square_coeff=0.5).sum()
        else:
            acceleration_penalty = soft_tanh_limit(self.joint_acc,
                -self.joint_qacc_max, self.joint_qacc_max, betas=(0.2, 0.2), square_coeff=0.5).sum()
        control_penalty = velocity_penalty + acceleration_penalty + position_penalty  # each term is bounded between [0, 1]
        control_penalty *= -self.control_penalty_coeff

        info = {"task_reward": task_reward,
                "velocity_penalty": -velocity_penalty*self.control_penalty_coeff,
                "position_penalty": -position_penalty*self.control_penalty_coeff,
                "acceleration_penalty": -acceleration_penalty*self.control_penalty_coeff,
                "control_penalty": control_penalty,
                "goal_dist": self.goal_dist,
                "guide_dist": np.sqrt(np.sum((self.box_pos - self.gripper_pos)**2))}
        reward = task_reward + control_penalty

        return state, reward, False, info


    def reset(self):
        if self._needs_rebuilding:
            raise RuntimeError("Env needs to be rebuilt after a parameter change! " \
                    "Most likely the randomizations are not defined properly. "\
                    "You need to call _rebuild_model() after every change to a "\
                    "parameter changed inside the XML.")

        self._reset_simulation()
        self._reset_model()

        return self._get_obs()

    def _reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel

        # Reset Panda to initial state
        qpos[self.arm_joint_index] = self.init_panda_qpos
        qvel[self.arm_joint_index] = self.init_panda_qvel

        self.set_state(qpos, qvel)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def _reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.data.ctrl[self.arm_act_index] = ctrl

        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:7],
                self.data.qvel.flat[:7]  + np.random.randn(7)*self.jvel_noise_stdev
            ]
        )

    def render(self, mode="human"):
        if mode == 'human':
            return self.mujoco_renderer.render(
                'human', camera_id=self.camera_id, camera_name=self.camera_name
            )
        return None

    def get_body_pos(self, body_name):
        """Return the cartesian position of a body frame"""
        return self.data.body(body_name).xpos

    def preprocess_action(self, action, clip=True):
        """Preprocess action from agent based on
        controller expected format"""
        if self.controller.ctrl_format == 'pos-vel-acc':
            """action: assumed acceleration in [-1, 1]""" 
            if clip:
                action = np.clip(action, -1, 1)
            acc = action * self.joint_qacc_max
            delta_vel = acc * self.dt

            joint_vel = self.joint_vel + np.random.randn(7)*0.0011

            # Clip velocity
            end_vel = joint_vel + delta_vel
            end_vel = np.clip(end_vel, self.joint_qvel_min, self.joint_qvel_max)
            true_acc = (end_vel - joint_vel)/self.dt

            return self.joint_pos, joint_vel, true_acc
        
        elif self.controller.ctrl_format == 'action':
            return action
        else:
            raise ValueError(f'Invalid format for control input: {self.controller.ctrl_format}')

    def _set_limits(self):
        # qpos limits
        self.joint_qpos_shift = np.array([0, 0, 0, 0, 0, -np.pi/2, -np.pi/4])
        # self.joint_qpos_shift = np.array([0, 0, 0, 0, 0, 0, 0])
        self.joint_qpos_min = np.array([-166, -101, -166, -176, -166, -1, -166])
        self.joint_qpos_min = self.joint_qpos_min / 180 * np.pi + self.joint_qpos_shift
        self.joint_qpos_max = np.array([166, 101, 166, -4, 166, 215, 166])
        self.joint_qpos_max = self.joint_qpos_max / 180 * np.pi + self.joint_qpos_shift

        # qvel limits
        joint_qvel_limits = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        self.joint_qvel_min = -joint_qvel_limits
        self.joint_qvel_max = joint_qvel_limits

        # qacc limits
        self.joint_qacc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])*self.qacc_factor

    @property
    def dt(self):
        """https://mujoco.readthedocs.io/en/latest/XMLreference.html#option
        :return: the simulation timestep
        """
        return self.model.opt.timestep

    @property
    def joint_pos(self):
        """
        :return: the joint position of the robot
        """
        return np.array(self.data.qpos[self.arm_joint_index])

    @property
    def joint_vel(self):
        """
        :return: the joint velocity of the robot
        """
        return np.array(self.data.qvel[self.arm_joint_index])

    @property
    def joint_acc(self):
        """
        :return: the joint acceleration of the robot
        """
        return np.array(self.data.qacc[self.arm_joint_index])

    @property
    def gripper_pos(self):
        """
        :return: the position of the robot end-effector.
            Visually, this position is marked with a small
            grey sphere, usually between the fingers of the robot.
        """
        site_id = self.model.site("panda_end_effector").id
        return np.array(self.data.site_xpos[site_id])

    def get_pair_friction(self, geom_name1, geom_name2):
        pair_id = self.get_contact_pair(geom_name1, geom_name2)
        pair_friction = self.model.pair_friction[pair_id]
        return pair_friction

    def get_contact_pair(self, geom_name1, geom_name2):
        # Find the proper geom ids
        geom_id1 = self.model.geom(geom_name1).id
        geom_id2 = self.model.geom(geom_name2).id

        # Find the right pair id
        pair_geom1 = self.model.pair_geom1
        pair_geom2 = self.model.pair_geom2
        pair_id = None
        for i, (g1, g2) in enumerate(zip(pair_geom1, pair_geom2)):
            if g1 == geom_id1 and g2 == geom_id2 \
               or g2 == geom_id1 and g1 == geom_id2:
                pair_id = i
                break
        if pair_id is None:
            raise KeyError("No contact between %s and %s defined."
                           % (geom_name1, geom_name2))
        return pair_id

    def set_armature(self, armature):
        self.model.dof_armature[self.arm_joint_index] = armature[:]

    def get_armature(self):
        return np.array(self.model.dof_armature[self.arm_joint_index])

    def set_joint_damping(self, damping):
        """Set the 7 joint dampings"""
        self.model.dof_damping[self.arm_joint_index] = damping[:]

    def get_joint_damping(self):
        return np.array(self.model.dof_damping[self.arm_joint_index])

    def set_joint_frictionloss(self, frictionloss):
        """Set the 7 joint frictionlosss"""
        self.model.dof_frictionloss[self.arm_joint_index] = frictionloss[:]

    def get_joint_frictionloss(self):
        return np.array(self.model.dof_frictionloss[self.arm_joint_index])