from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb

import gym
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class PartialInfoExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, observation_mask: List[int]) -> None:
        """Filter observations according to the given mask of indices.
            Used for giving less information to the policy, while full information
            to the Q function.

            observation_mask: list
                              mask of indices to RETAIN in the observation vector            
        """
        assert isinstance(observation_mask, list) and observation_mask is not None
        assert np.all(np.array(observation_mask) >= 0) and np.all(np.array(observation_mask) < get_flattened_obs_dim(observation_space)), 'mask indices must be contained within the length of the obs vector dimension'

        original_ob_length = get_flattened_obs_dim(observation_space)
        features_dim = len(observation_mask)
        super().__init__(observation_space, features_dim)
        
        self.ob_mask = observation_mask
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs = self.flatten(observations)
        masked_obs = obs[:, self.ob_mask]  # mask observations in the obs dimension
        return masked_obs