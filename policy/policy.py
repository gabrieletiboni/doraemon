"""
    Interface to sb3 APIs for policy training and evaluation
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import sys
import numpy as np
import torch
import pdb
import os

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC

from .callbacks import WandbRecorderCallback
from .AsymmetricSACPolicy import AsymmetricSACPolicy
from .PartialInfoExtractor import PartialInfoExtractor 


class Policy:
    
    def __init__(self,
                 algo=None,
                 env=None,
                 lr=0.0003,
                 gamma=0.99,
                 device='cpu',
                 seed=None,
                 load_from_pathname=None,
                 actor_obs_mask: List[int] = None,
                 critic_obs_mask: List[int] = None,
                 gradient_steps=-1):
        """
            Policy class that handles training and making all required networks.
            It provides an easy-to-use interface to sb3 APIs, with wandb support, automatic
            best model returned, and more under the hood.

            :param actor_obs_mask: observation mask for the policy, while Q function has
                                    full observation information. Used for off-policy algorithms only.
            :param critic_obs_mask: observation mask for the critic. used to filter out history in the obs
                                    and use just the low dimensional xi vector instead.
        """
        self.seed = seed
        self.device = device
        self.env = env
        self.algo = algo
        self.actor_obs_mask = actor_obs_mask
        self.critic_obs_mask = critic_obs_mask
        self.gradient_steps = gradient_steps

        if load_from_pathname is None:
            self.model = self.create_model(algo, lr=lr, gamma=gamma)
        else:
            self.model = self.load_model(algo, load_from_pathname)

        return
    
    def create_model(self, algo, lr, gamma):
        if algo == 'ppo':
            policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                                 net_arch=dict(pi=[128, 128], vf=[128, 128]))
            model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, learning_rate=lr, gamma=gamma, verbose=0, seed=self.seed, device=self.device)

        elif algo == 'sac':
            policy_class, policy_kwargs = self._get_policy_params()
            policy_kwargs = {
                                **policy_kwargs,
                                'activation_fn': torch.nn.Tanh,
                                'net_arch': dict(pi=[128, 128], qf=[128, 128])
                            }
            model = SAC(policy_class,
                        self.env,
                        policy_kwargs=policy_kwargs,
                        learning_rate=lr,
                        gamma=gamma,
                        gradient_steps=self.gradient_steps,  # -1 means as many as env.num_envs
                        verbose=0,
                        seed=self.seed,
                        device=self.device)
        else:
            raise ValueError(f"RL Algo not supported: {algo}")

        return model

    def _get_policy_params(self):
        """Return policy_class and policy_kwargs depending
        on whether or not to use the custom implementation for
        asymmetric info on policy and q function
        """
        policy_kwargs = dict()
        if self.actor_obs_mask is not None:  # use custom AsymmetricSACPolicy with custom features_extractor PartialInfoExtractor
            policy_class = AsymmetricSACPolicy
            policy_kwargs = dict(
                features_extractor_class_actor=PartialInfoExtractor,
                features_extractor_actor_kwargs=dict(observation_mask=self.actor_obs_mask),
                share_features_extractor=False
            )

        if self.critic_obs_mask is not None:
            policy_class = AsymmetricSACPolicy
            policy_kwargs = {**policy_kwargs,
                             **dict(
                                features_extractor_class_critic=PartialInfoExtractor,
                                features_extractor_critic_kwargs=dict(observation_mask=self.critic_obs_mask),
                                share_features_extractor=False
                             )
                            }

        if self.actor_obs_mask is None and self.critic_obs_mask is None:
            policy_class, policy_kwargs = "MlpPolicy", dict()

        return policy_class, policy_kwargs

    def load_model(self, algo, pathname):
        if algo == 'ppo':
            model = PPO.load(pathname, env=self.env, device=self.device)
        elif algo == 'sac':
            model = SAC.load(pathname, env=self.env, device=self.device)
        else:
            raise ValueError(f"RL Algo not supported: {algo}")
        return model

    def train(self,
              timesteps=1000,
              stopAtRewardThreshold=False,
              reward_threshold=None,
              n_eval_episodes=50,
              eval_freq=1000,
              best_model_save_path=None,
              return_best_model=True,
              wandb_loss_suffix="",
              verbose=0,
              render_eval=False,
              eval_env=None,
              custom_callbacks=None,
              before_final_eval_cb=None,
              after_final_eval_cb=None,
              disable_eval=False):
        """Train a model

            1. Setup callbacks
            2. Train model
            3. Find best model and return it
        """

        if self.model.get_env().env_method('get_reward_threshold')[0] is not None and stopAtRewardThreshold:
            reward_threshold = reward_threshold if reward_threshold is not None else self.model.get_env().env_method('get_reward_threshold')[0]
            stop_at_reward_threshold = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        else:
            stop_at_reward_threshold = None

        wandb_recorder = WandbRecorderCallback(eval_freq=eval_freq, wandb_loss_suffix=wandb_loss_suffix, verbose=verbose) # Plot stuff on wandb
        eval_callback = EvalCallback(self.env if eval_env is None else eval_env,
                                     best_model_save_path=best_model_save_path,
                                     eval_freq=eval_freq,
                                     n_eval_episodes=n_eval_episodes,
                                     deterministic=True,
                                     callback_after_eval=wandb_recorder,
                                     callback_on_new_best=stop_at_reward_threshold,
                                     verbose=verbose,
                                     render=render_eval)

        # Additionally test policy on other environments
        if custom_callbacks is not None:
            assert isinstance(custom_callbacks, list)
            eval_callback = CallbackList([eval_callback] + custom_callbacks)

        if disable_eval:
            eval_callback = None

        self.model.learn(total_timesteps=timesteps, callback=eval_callback)

        if return_best_model:
            # Find best model among last and best
            
            reward_final, std_reward_final = self.eval(n_eval_episodes=n_eval_episodes, eval_env=(eval_env if eval_env is not None else None))

            assert os.path.exists(os.path.join(best_model_save_path, "best_model.zip")), "best_model.zip hasn't been saved because too few evaluations have been performed. Check --eval_freq and -t"
            best_model = self.load_model(self.algo, os.path.join(best_model_save_path, "best_model.zip"))
            reward_best, std_reward_best = evaluate_policy(best_model, best_model.get_env(), n_eval_episodes=n_eval_episodes)

            if reward_final > reward_best:
                best_policy = self.state_dict()
                best_mean_reward, best_std_reward = reward_final, std_reward_final
                which_one = 'final'
            else:
                best_policy = best_model.policy.state_dict()
                best_mean_reward, best_std_reward = reward_best, std_reward_best
                which_one = 'best'

            info = {'which_one': which_one}

            return best_mean_reward, best_std_reward, best_policy, info
        else:
            # Return final policy
            if before_final_eval_cb is not None:
                before_final_eval_cb()
            reward_final, std_reward_final = self.eval(n_eval_episodes=n_eval_episodes, eval_env=(eval_env if eval_env is not None else None))
            if after_final_eval_cb is not None:
                after_final_eval_cb()
            final_policy = self.state_dict()

            info = {'which_one': 'final'}

            return reward_final, std_reward_final, final_policy, info

    def eval(self, n_eval_episodes=50, render=False, eval_env=None):
        mean_reward, std_reward = evaluate_policy(self.model, (self.model.get_env() if eval_env is None else eval_env), n_eval_episodes=n_eval_episodes, render=render)
        return mean_reward, std_reward

    def predict(self, state, deterministic=False):
        return self.model.predict(state, deterministic=deterministic)

    def state_dict(self):
        return self.model.policy.state_dict()

    def save_state_dict(self, pathname):
        torch.save(self.state_dict(), pathname)

    def load_state_dict(self, path_or_state_dict):
        if type(path_or_state_dict) is str:
            self.model.policy.load_state_dict(torch.load(path_or_state_dict, map_location=torch.device(self.device)), strict=True)
        else:
            self.model.policy.load_state_dict(path_or_state_dict, strict=True)

    def save_full_state(self, pathname):
        self.model.save(pathname)

    def load_full_state(self, pathname):
        raise ValueError('Use the constructor with load_from_pathname parameter')
        pass