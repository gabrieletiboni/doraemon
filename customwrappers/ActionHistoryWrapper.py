"""General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
    for stacking previous actions in the current observation vector
"""
import pdb

import gym
import numpy as np

class ActionHistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_len, valid_dim=False):
        """
            Augments the observation with
            a stack of the previous "history_len" actions
            taken.

            valid_dim : bool
                        if False, at the beginning of the episode, zero-valued actions
                            are used.
                        if True, an additional binary valid code is used as input to indicate whether
                        previous actions are valid or not (beginning of the episode).
        """
        super().__init__(env)
        assert env.action_space.sample().ndim == 1, 'Actions are assumed to be flat on one-dim vector'
        assert valid_dim == False, 'valid encoding has not been implemented yet.'

        self.history_len = history_len
        self.actions_buffer = np.zeros((history_len, env.action_space.shape[0]), dtype=np.float32)

        # Modify the observation space to include the history buffer
        obs_space = env.observation_space
        action_stack_low = np.repeat(env.action_space.low, history_len)
        action_stack_high = np.repeat(env.action_space.high, history_len)
        low = np.concatenate([obs_space.low.flatten(), action_stack_low], axis=0)
        high = np.concatenate([obs_space.high.flatten(), action_stack_high], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.actions_buffer.fill(0)
        return self._stack_actions_to_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.actions_buffer[:-1] = self.actions_buffer[1:]
        self.actions_buffer[-1] = action
        obs = self._stack_actions_to_obs(obs)
        return obs, reward, done, info

    def _stack_actions_to_obs(self, obs):
        obs = np.concatenate([obs.flatten(), self.actions_buffer.flatten()], axis=0)
        return obs