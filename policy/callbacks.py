import pdb

import wandb
import numpy as np
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class WandbRecorderCallback(BaseCallback):
    """
    A custom callback that allows to print stuff on wandb after every evaluation

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_freq=None, wandb_loss_suffix="", verbose=0):
        super(WandbRecorderCallback, self).__init__(verbose)

        self.wandb_loss_suffix = wandb_loss_suffix
        self.wandb_extra_metrics_min = None
        self.wandb_extra_metrics_max = None

        self.verbose = verbose


    def _on_step(self) -> bool:
        """
        This method is called as a child callback of the `EventCallback`,
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.

        Print stuff on wandb
        """
        last_mean_reward = self.parent.last_mean_reward
        
        current_timestep = self.num_timesteps  # this number is already multiplied by the number of parallel envs
        wandb.log({"train_mean_reward"+self.wandb_loss_suffix: last_mean_reward, "timestep": current_timestep})

        if self.verbose >= 1:
            print(f'======\nEval policy: timestep {self.num_timesteps} | train_mean_reward {last_mean_reward}', end='')

        ### Plot extra metrics (assumes env of type VecEnv)
        wandb_extra_metrics = None
        if isinstance(self.training_env, VecEnv):
            if np.all(self.training_env.has_attr('wandb_extra_metrics')):
                wandb_extra_metrics = self.training_env.get_attr('wandb_extra_metrics')[0]
                if self.wandb_extra_metrics_min is None or self.wandb_extra_metrics_max is None:  # Initialize min and max values
                    self.wandb_extra_metrics_min = {v: np.inf for k, v in wandb_extra_metrics.items()}
                    self.wandb_extra_metrics_max = {v:-np.inf for k, v in wandb_extra_metrics.items()}

        if wandb_extra_metrics is not None:
            for k, v in wandb_extra_metrics.items():
                mean_value = np.mean(self.training_env.get_attr(k))  # mean across parallel envs
                wandb.log({v: mean_value, "timestep": current_timestep})
                if self.verbose >= 1:
                    print(f' | {v} {mean_value}', end='')
                if mean_value < self.wandb_extra_metrics_min[v]:  # update min
                    self.wandb_extra_metrics_min[v] = mean_value
                    wandb.run.summary[f"{v}_MIN"] = mean_value

                if mean_value > self.wandb_extra_metrics_max[v]:  # update max
                    self.wandb_extra_metrics_max[v] = mean_value
                    wandb.run.summary[f"{v}_MAX"] = mean_value

        if self.verbose >= 1:
            print()

        return True