import pdb
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

from dr_envs.random_env import RandomEnv

class RandomPlane(RandomEnv):
    def __init__(self, difficulty='hard'):
        RandomEnv.__init__(self)

        # Define the observation space (box position)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Define the action space (applied horizontal force)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Initialize the box position and other variables
        self.box_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.box_vel = np.array([0.0, 0.0], dtype=np.float32)

        if difficulty == 'hard':
            self.init_box_pos_distr = [-0.45, 0.45]
            self.init_box_vel_distr = [-0.1, 0.1]
        elif difficulty == 'easy':
            self.init_box_pos_distr = [-0.05, 0.05]
            self.init_box_vel_distr = [-0.0, 0.0] 
        else:
            raise ValueError(f'Difficulty value is not supported: {difficulty}')

        self.gravity = 9.81
        self.timestep = 0.05
        self.box_mass = 1.
        self.max_force = 3.
        self.theta = 0  # Nominal angle (flat plane)

        # Observation space
        high = np.array([np.finfo(np.float32).max,  # pos_robot
                         np.finfo(np.float32).max], # vel_robot
                         dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action space
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        self.dyn_ind_to_name = {0: 'angle'}

        self.seed()
        self.viewer = None

        self.task_dim = len(self.dyn_ind_to_name.keys())
        self.original_task = np.array([self.theta])
        self.nominal_values = np.copy(self.original_task)
        self.min_task = np.zeros(self.task_dim)
        self.max_task = np.zeros(self.task_dim)
        self.mean_task = np.zeros(self.task_dim)
        self.stdev_task = np.zeros(self.task_dim)

        self.preferred_lr = None
        self.reward_threshold = 0  # temp

        self.wandb_extra_metrics = {'time_at_the_center': 'time_at_the_center'}
        self.success_metric = 'time_at_the_center'
        self.time_at_the_center = 0.
        self.center_is_within = 0.05  # distance from center to be considered "in the center"

        self.verbose = 0

    def reset(self):
        # Sample new dynamics
        if self.dr_training:
            self.set_random_task()

        # Reset box position and velocity
        self.box_pos = np.array([np.random.uniform(low=self.init_box_pos_distr[0], high=self.init_box_pos_distr[1]), 0.0], dtype=np.float32)
        self.box_vel = np.array([np.random.uniform(low=self.init_box_vel_distr[0], high=self.init_box_vel_distr[1]), 0.0], dtype=np.float32)

        # Reset time at the center
        self.time_at_the_center = 0.

        return self.box_pos

    def step(self, action):
        # Update the box position based on the applied force and gravity
        input_force = action[0] * self.max_force

        acceleration = np.array([
                            (input_force/self.box_mass - np.sin(self.theta) * self.gravity),
                            0
                       ], dtype=np.float32)

        self.box_vel = self.box_vel +  acceleration*self.timestep
        self.box_pos = self.box_pos + self.box_vel*self.timestep

        has_fallen = self.has_fallen(self.box_pos)
        reward = self._get_reward(self.box_pos) - (self.fall_penalty() if has_fallen else 0.)
        done = has_fallen
        info = {'is_at_the_center': self.is_at_the_center(self.box_pos)}

        if self.is_at_the_center(self.box_pos):
            self.time_at_the_center += 1.
        else:
            self.time_at_the_center = 0.

        return self._get_state(), reward, done, info

    def _get_state(self):
        return np.array([self.box_pos[0], self.box_vel[0]], dtype=np.float32)

    def _get_reward(self, x):
        return -x[0]**2

    def has_fallen(self, x):
        return x[0] > 0.5 or x[0] < -0.5

    def fall_penalty(self):
        return 1.

    def is_at_the_center(self, x):
        return x[0] <= self.center_is_within and x[0] >= -self.center_is_within

    def from_robot_to_world(self, x):
        assert x.shape[0] == 2
        return self.rot_robot_to_world @ x

    def from_world_to_robot(self, x):
        assert x.shape[0] == 2
        return np.linalg.inv(self.rot_robot_to_world) @ x

    @property
    def rot_robot_to_world(self):
        return np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
    
    def render(self, mode='human'):
        """Render the scene"""
        plt.figure()
        plt.plot(self.from_robot_to_world(self.box_pos)[0], self.from_robot_to_world(self.box_pos)[1], 'bo')
        plt.plot([-np.cos(self.theta)/2, np.cos(self.theta)/2], [-np.sin(self.theta)/2, np.sin(self.theta)/2], 'r')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Inclined Plane with Box')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.8)
        plt.close()

    def get_task(self):
        return np.array([self.theta])

    def set_task(self, *task):
        self.theta = task[0]

    def get_search_bounds_mean(self, index):
        """Get search bounds for the mean of the parameters optimized
        """
        search_bounds_mean = {
               'angle': (-np.pi/2, np.pi/2),
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        lowest_value = {
                    'angle': -np.pi/2,
        }
        return lowest_value[self.dyn_ind_to_name[index]]

    def get_task_upper_bound(self, index):
        """Returns lowest feasible value for each dynamics

        Used for resampling unfeasible values during domain randomization
        """
        upper_value = {
                    'angle': np.pi/2,
        }
        return upper_value[self.dyn_ind_to_name[index]]

    def set_verbosity(self, verbose):
        self.verbose = verbose


gym.envs.register(
    id="RandomPlane-v0",
    entry_point="%s:RandomPlane" % __name__,
    max_episode_steps=50,
    kwargs={}
)

gym.envs.register(
    id="RandomPlaneEasy-v0",
    entry_point="%s:RandomPlane" % __name__,
    max_episode_steps=50,
    kwargs={'difficulty': 'easy'}
)