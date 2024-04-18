import gym
import numpy as np

class DynamicsInObs(gym.ObservationWrapper):
    def __init__(self, env, dynamics_mask=None):
        """
            Stack the current env dynamics to the env observation vector

            dynamics_mask: list of int
                           indices of dynamics to randomize, i.e. to condition the network on
        """
        super().__init__(env)

        if dynamics_mask is not None:
            self.dynamics_mask = np.array(dynamics_mask)
            task_dim = env.get_task()[self.dynamics_mask].shape[0]
        else:  # All dynamics are used
            task_dim = env.get_task().shape[0]
            self.dynamics_mask = np.arange(task_dim)

        self.nominal_values = env.get_task()[self.dynamics_mask].copy()  # used for normalizing dynamics values

        obs_space = env.observation_space
        low = np.concatenate([obs_space.low.flatten(), np.repeat(-np.inf, task_dim)], axis=0)
        high = np.concatenate([obs_space.high.flatten(), np.repeat(np.inf, task_dim)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def observation(self, obs):
        norm_dynamics = self.get_task()[self.dynamics_mask] - self.nominal_values
        obs = np.concatenate([obs.flatten(), norm_dynamics], axis=0)
        return obs