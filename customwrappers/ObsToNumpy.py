import gym
import numpy as np

class ObsToNumpy(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.asarray(obs)