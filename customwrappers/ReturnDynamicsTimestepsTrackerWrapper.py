"""General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
    for keeping track of dynamics and associated returns when training with DORAEMON.
"""
import pdb

import gym
import numpy as np
import torch

class ReturnDynamicsTimestepsTrackerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        # Override wrapped env set_random_task method
        self.overrides['set_random_task'] = self.doraemon_custom_set_random_task
        self.buffer = []  # keep track of (dynamics, return) tuples
        self.succ_metric_buffer = []  # buffer with metric used for measuring success
        self.timesteps = []

        self.cum_reward = 0
        self.curr_timesteps = 0
        self.curr_episode_dynamics = None
        self.ready_to_update_buffer = False
        self.expose_episode_stats = False
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        if self.expose_episode_stats:
            # Compute return over time
            self.cum_reward += reward
            self.curr_timesteps += 1

            if done and self.ready_to_update_buffer:
                self._update_buffer(episode_dynamics=self.curr_episode_dynamics, episode_return=self.cum_reward, timesteps=self.curr_timesteps)

                if hasattr(self.env, 'success_metric'):
                    self._update_succ_metric_buffer(succ_metric=getattr(self.env, self.success_metric))

                self.cum_reward = 0
                self.curr_timesteps = 0
                self.ready_to_update_buffer = False

        return next_state, reward, done, info

    def doraemon_custom_set_random_task(self):
        """Keep track of sampled task and its return"""
        task = self.sample_task()
        self.set_task(*task)

        if self.expose_episode_stats:
            self.cum_reward = 0
            self.curr_timesteps = 0
            self.curr_episode_dynamics = np.array(task, dtype=np.float64)
            self.ready_to_update_buffer = True

    def _update_buffer(self, episode_dynamics, episode_return, timesteps):
        self.buffer.append({'dynamics': episode_dynamics, 'return': episode_return, 'timesteps': timesteps})

    def _update_succ_metric_buffer(self, succ_metric):
        self.succ_metric_buffer.append(succ_metric)

    def reset_buffer(self):
        self.buffer = []
        self.succ_metric_buffer = []

    def get_buffer(self):
        return self.buffer

    def get_succ_metric_buffer(self):
        return self.succ_metric_buffer

    def set_expose_episode_stats(self, flag):
        self.expose_episode_stats = flag