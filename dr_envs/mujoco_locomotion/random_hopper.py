"""Implementation of the Hopper environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/hopper/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from dr_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class RandomHopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, noisy=False):
        self.original_lengths = np.array([.4, .45, 0.5, .39])
        self.model_args = {"size": list(self.original_lengths)}

        MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

        # Define randomized dynamics
        self.dyn_ind_to_name = {0: 'torsomass', 1: 'thighmass', 2: 'legmass', 3: 'footmass',
                                4: 'damping0', 5: 'damping1', 6: 'damping2', 7: 'friction'}

        default_task = self.get_default_task()
        self.set_task(*default_task)
        self.original_masses = np.copy(self.get_task())
        self.nominal_values = np.concatenate([self.original_masses])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.preferred_lr = 0.0005
        self.reward_threshold = 1600
    

    def get_default_task(self):
        mean_of_search_bounds = np.array([(self.get_search_bounds_mean(i)[0] + self.get_search_bounds_mean(i)[1])/2 for i in range(len(self.dyn_ind_to_name.keys()))])
        return mean_of_search_bounds

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'torsomass': (0.1, 10.0),
               'thighmass': (0.1, 10.0),
               'legmass': (0.1, 10.0),
               'footmass': (0.1, 10.0),
               'damping0': (0.1, 3.),
               'damping1': (0.1, 3.),
               'damping2': (0.1, 3.),
               'friction': (0.1, 3.)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torsomass': 0.001,
                    'thighmass': 0.001,
                    'legmass': 0.001,
                    'footmass': 0.001,
                    'damping0': 0.05,
                    'damping1': 0.05,
                    'damping2': 0.05,
                    'friction': 0.01
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:] )
        damping = np.array( self.sim.model.dof_damping[3:] )
        friction = np.array( [self.sim.model.pair_friction[0, 0]] )
        return np.concatenate([masses, damping, friction])

    def set_task(self, *task):
        self.sim.model.body_mass[1:] = task[:4]
        self.sim.model.dof_damping[3:] = task[4:7]  # damping on the three actuated joints
        self.sim.model.pair_friction[0, :2] = np.repeat(task[7], 2)


    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
            # np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

        return obs

    def reset_model(self):
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
            
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def set_sim_state(self, mjstate):
        return self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="RandomHopper-v0",
        entry_point="%s:RandomHopperEnv" % __name__,
        max_episode_steps=500
)