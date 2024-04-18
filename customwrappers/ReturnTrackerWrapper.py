"""General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
    for keeping track of experienced returns while training/evaluating.
"""
import pdb

import gym

class ReturnTrackerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.buffer = []  # keep track of experienced returns
        self.succ_metric_buffer = []  # buffer with metric used for measuring success
        self.exposed_cum_reward = 0
        self.ready_to_update_buffer = False
        self.expose_episode_stats = False
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        if self.expose_episode_stats:
            # Compute return over time
            self.exposed_cum_reward += reward

            if done and self.ready_to_update_buffer:
                self._update_buffer(episode_return=self.exposed_cum_reward)

                if hasattr(self.env, 'success_metric'):
                    self._update_succ_metric_buffer(succ_metric=getattr(self.env, self.success_metric))

                self.exposed_cum_reward = 0
                self.ready_to_update_buffer = False


        return next_state, reward, done, info

    def reset(self):
        ob = self.env.reset()

        if self.expose_episode_stats:
            self.exposed_cum_reward = 0
            self.ready_to_update_buffer = True
        return ob

    def _update_buffer(self, episode_return):
        self.buffer.append(episode_return)

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

    def get_expose_episode_stats(self):
        return self.expose_episode_stats