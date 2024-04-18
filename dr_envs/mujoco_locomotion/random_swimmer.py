"""Implementation of the Swimmer environment supporting
domain randomization optimization.

For all details: https://www.gymlibrary.ml/environments/mujoco/swimmer/
"""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from dr_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class RandomSwimmerEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, noisy=False):
        self.original_lengths = np.array([.1, .1, .1])
        self.original_viscosity = 0.1
        self.model_args = {"size": list(self.original_lengths), 'viscosity': self.original_viscosity}

        self.noisy = noisy
        self.noise_level = 1e-4

        MujocoEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.nominal_values = np.concatenate([self.original_masses, self.original_lengths, [self.original_viscosity]])
        self.task_dim = self.nominal_values.shape[0]
        self.current_lengths = np.array(self.original_lengths)
        self.current_viscosity = self.original_viscosity

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'mass0', 1: 'mass1', 2: 'mass2', 3: 'size0', 4: 'size1', 5: 'size2', 6: 'viscosity'}

        self.preferred_lr = 0.0005
        self.reward_threshold = 1750
        

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'mass0': (10, None),
               'mass1': (10, None),
               'mass2': (10, None),
               'size0': (0.03, None),
               'size1': (0.03, None),
               'size2': (0.03, None),
               'viscosity': (0.01, None)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'mass0': 0.001,
                    'mass1': 0.001,
                    'mass2': 0.001,
                    'size0': 0.01,
                    'size1': 0.01,
                    'size2': 0.01,
                    'viscosity': 0.005
        }

        return lowest_value[self.dyn_ind_to_name[index]]


    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:] )
        return np.concatenate((masses, self.current_lengths, [self.current_viscosity]))

    def set_task(self, *task):
        self.current_lengths = np.array(task[len(self.original_masses):len(self.original_masses)+len(self.original_lengths)])
        self.current_viscosity = task[-1]
        self.model_args = {"size": list(self.current_lengths), 'viscosity': self.current_viscosity}
        self.build_model()
        self.sim.model.body_mass[1:] = task[:len(self.original_masses)]
        return


    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()

        return (
            ob,
            reward,
            False,
            dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        obs = np.concatenate([qpos.flat[2:], qvel.flat])

        if self.noisy:
            obs += np.sqrt(self.noise_level)*np.random.randn(obs.shape[0])

        return obs

    def reset_model(self):
        # Before potentially re-building the model
        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


gym.envs.register(
        id="RandomSwimmer-v0",
        entry_point="%s:RandomSwimmerEnv" % __name__,
        max_episode_steps=500
)

gym.envs.register(
        id="RandomSwimmerNoisy-v0",
        entry_point="%s:RandomSwimmerEnv" % __name__,
        max_episode_steps=500,
        kwargs={"noisy": True}
)