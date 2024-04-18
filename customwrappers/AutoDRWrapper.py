"""General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
    for handling ADR boundary sampling probability.
"""
import pdb

import gym
import numpy as np

class AutoDRWrapper(gym.Wrapper):
    def __init__(self, env, bound_sampling_prob):
        super().__init__(env)
        self.env = env
        self.bound_sampling_prob = bound_sampling_prob

        # Override wrapped env set_random_task method
        self.overrides['set_random_task'] = self.adr_custom_set_random_task
        self.buffers = [ [[], []] for i in range(self.task_dim)]
        self.succ_metric_buffers = [ [[], []] for i in range(self.task_dim)]

        self.cum_reward = 0
        self.adr_eval_mode = False
        self.ready_to_update_buffers = False
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # Handle AutoDR boundary sampling evaluation
        if self.adr_eval_mode:
            self.cum_reward += reward

            if done and self.ready_to_update_buffers:
                self._update_buffers()

                if hasattr(self.env, 'success_metric'):
                    self._update_succ_metric_buffers(succ_metric=getattr(self.env, self.success_metric))

                self.cum_reward = 0
                self.ready_to_update_buffers = False

        return next_state, reward, done, info

    def adr_custom_set_random_task(self):
        """Sample and set random parameters"""
        task = np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

        p = np.random.uniform()
        if p < self.bound_sampling_prob:
            # ADR Eval worker
            self.sample_dim = np.random.choice(self.task_dim)
            self.low_or_high = 0 if np.random.uniform() < 0.5 else 1

            if self.low_or_high == 0:
                # fix one dim to low bound
                task[self.sample_dim] = self.min_task[self.sample_dim]
            else:
                # fix one dim to high bound
                task[self.sample_dim] = self.max_task[self.sample_dim]

            self.set_task(*task)
            # print(f"ADR worker (dim:{self.sample_dim},{('low' if self.low_or_high == 0 else 'high')}). Task:", self.get_task())

            self.adr_eval_mode = True
            self.ready_to_update_buffers = True
            self.cum_reward = 0

        else:
            # Rollout Worker 
            self.adr_eval_mode = False
            self.set_task(*task)
            # print('Rollout worker. Task:', self.get_task())

    def _update_buffers(self):
        self.buffers[self.sample_dim][self.low_or_high].append(self.cum_reward)

    def _update_succ_metric_buffers(self, succ_metric):
        self.succ_metric_buffers[self.sample_dim][self.low_or_high].append(succ_metric)

    def reset_buffer(self, dim, low_or_high):
        self.buffers[dim][low_or_high] = []
        self.succ_metric_buffers[dim][low_or_high] = []

    def get_buffers(self):
        return self.buffers

    def get_succ_metric_buffers(self):
        return self.succ_metric_buffers