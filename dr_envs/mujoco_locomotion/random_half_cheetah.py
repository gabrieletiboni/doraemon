"""Implementation of the HalfCheetah environment supporting
domain randomization optimization.

Randomizations:
    - 7 mass links
    - 1 friction coefficient (sliding)

For all details: https://www.gymlibrary.ml/environments/mujoco/half_cheetah/
"""
import numpy as np
import gym
from gym import utils
from dr_envs.mujoco_locomotion.jinja_mujoco_env import MujocoEnv
from copy import deepcopy
import pdb

class RandomHalfCheetah(MujocoEnv, utils.EzPickle):
    def __init__(self, noisy=False):
        self.original_lengths = np.array([1., .15, .145, .15, .094, .133, .106, .07])
        self.model_args = {"size": list(self.original_lengths)}


        self.noisy = noisy
        # Rewards:
        #   noiseless: 5348
        #   1e-5: 5273 +- 75
        #   1e-4: 4793 +- 804
        #   1e-3: 2492 +- 472
        #   1e-2: 
        self.noise_level = 1e-4


        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])
        self.original_friction = np.array([0.4])
        self.nominal_values = np.concatenate([self.original_masses, self.original_friction])
        self.task_dim = self.nominal_values.shape[0]

        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)

        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.dyn_ind_to_name = {0: 'torso', 1: 'bthigh', 2: 'bshin', 3: 'bfoot', 4: 'fthigh', 5: 'fshin', 6: 'ffoot', 7: 'friction' }

        self.preferred_lr = 0.0005 # --algo Sac -t 5M
        self.reward_threshold = 4500

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized"""
        search_bounds_mean = {
               'torso': (0.1, 10.0),
               'bthigh': (0.1, 10.0),
               'bshin': (0.1, 10.0),
               'bfoot': (0.1, 10.0),
               'fthigh': (0.1, 10.0),
               'fshin': (0.1, 10.0),
               'ffoot': (0.1, 10.0),
               'friction': (0.1, 2.0),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'torso': 0.1,
                    'bthigh': 0.1,
                    'bshin': 0.1,
                    'bfoot': 0.1,
                    'fthigh': 0.1,
                    'fshin': 0.1,
                    'ffoot': 0.1,
                    'friction': 0.02,
        }

        return lowest_value[self.dyn_ind_to_name[index]]

    def get_task(self):
        masses = np.array( self.sim.model.body_mass[1:] )
        friction = np.array( self.sim.model.pair_friction[0,0] )
        task = np.append(masses, friction)
        return task

    def set_task(self, *task):
        # self.current_lengths = np.array(task[-len(self.original_lengths):])
        # self.model_args = {"size": list(self.current_lengths)}
        # self.build_model()
        # self.sim.model.body_mass[1:] = task[:-len(self.original_lengths)]

        self.sim.model.body_mass[1:] = task[:-1]
        self.sim.model.pair_friction[0:2,0:2] = task[-1]


    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

        if self.noisy:
            obs += np.sqrt(self.noise_level)*np.random.randn(obs.shape[0])

        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        if self.dr_training:
            self.set_random_task() # Sample new dynamics

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_sim_state(self, mjstate):
        mjstate = self.sim.set_state(mjstate)

    def get_sim_state(self):
        return self.sim.get_state()


gym.envs.register(
        id="RandomHalfCheetah-v0",
        entry_point="%s:RandomHalfCheetah" % __name__,
        max_episode_steps=500
)

gym.envs.register(
        id="RandomHalfCheetahNoisy-v0",
        entry_point="%s:RandomHalfCheetah" % __name__,
        max_episode_steps=500,
        kwargs={"noisy": True}
)