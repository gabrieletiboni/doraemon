"""Train a policy using Uniform Domain Randomization.

    Examples:
        (DEBUG)
            python train_udr.py --wandb disabled --env RandomContinuousInvertedCartPoleEasy-v0 -t 1000 --algo sac --eval_freq 500 --eval_episodes 1 --test_episodes 1 --seed 42 --dr_percentage 0.95 --verbose 1 --debug

        (See readme.md for reproducing paper results)
"""
from pprint import pprint
import argparse
import gc
import pdb
import sys
import socket
import os

import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import wandb
from stable_baselines3.common.env_util import make_vec_env

import dr_envs
from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from utils.utils import *
from utils.gym_utils import *
from policy.policy import Policy


def main():
    args.eval_freq = max(args.eval_freq // args.now, 1)   # Making eval_freq behave w.r.t. global timesteps, so it follows --timesteps convention
    torch.set_num_threads(max(5, args.now))  # hard-coded for now. Avoids taking up all CPUs when parallelizing with multiple environments and processes on hephaestus

    assert args.dr_percentage <= 1 and args.dr_percentage >= 0
    assert args.env is not None
    assert args.test_env is None, 'source and target domains should be the same. As of right now, test_env is used to test the policy on the final target DR distribution'
    if args.test_env is None:
        args.test_env = args.env

    gt_task = gym.make(args.env, **env_kwargs).get_task()  # ground truth dynamics parameters (static vector)

    if args.rand_all_but is not None:  # args.rand_all_but overwrites args.rand_only
        args.rand_only = np.arange(len(gt_task)).tolist()
        del args.rand_only[args.rand_all_but]
    

    ### Configs and Wandb
    random_string = get_random_string(5)
    run_name = "UDR_"+ args.algo +'_seed'+str(args.seed)+'_'+random_string
    print(f'========== RUN_NAME: {run_name} ==========')
    pprint(vars(args))
    set_seed(args.seed)
    wandb.init(config=vars(args),
               project="doraemon-rl",
               group="UDR_"+str(args.env if args.group is None else args.group),
               name=run_name,
               save_code=True,
               tags=None,
               notes=args.notes,
               mode=args.wandb)

    run_path = "runs/"+str(args.env)+"/"+get_run_name(args)+"_"+random_string+"/"
    print('Run path:', run_path)
    create_dirs(run_path)
    save_config(vars(args), run_path)
    wandb.config.path = run_path
    wandb.config.hostname = socket.gethostname()



    ### Get training uniform distribution for DR
    print('Ground truth task:', gt_task)
    lower_bounds = np.zeros(len(gt_task)) if is_locomotion_env(args.env) else None  # use zeros as lower_bounds for locomotion envs params
    training_bounds = gym.make(args.env, **env_kwargs).get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                                                    nominal_values=gt_task,
                                                                                    lower_bounds=lower_bounds,
                                                                                    dyn_mask=args.rand_only)
    print('Training bounds:', np.round(training_bounds.reshape(-1,2),2))



    ### Actor & Critic input observation masks for asymmetric information
    actor_obs_mask, critic_obs_mask = get_actor_critic_obs_masks(args)



    ### Training
    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args}, env_kwargs=env_kwargs)
    env.set_dr_distribution(dr_type='uniform', distr=training_bounds)
    env.set_dr_training(True)
    eff_lr = get_learning_rate(args, env)
    policy = Policy(algo=args.algo,
                    env=env,
                    lr=eff_lr,
                    gamma=args.gamma,
                    device=args.device,
                    seed=args.seed,
                    actor_obs_mask=actor_obs_mask,
                    critic_obs_mask=critic_obs_mask,
                    gradient_steps=args.gradient_steps)

    eval_env = None
    render_eval = False
    if args.render_eval:
        assert args.stack_history is None and not args.dyn_in_obs, 'render env is created without the dedicated asymmetric info wrappers. Implement this in gym_utils.py'
        eval_env = make_rendering_env(args.env, env_kwargs=env_kwargs)
        eval_env.set_dr_distribution(dr_type='uniform', distr=training_bounds)
        eval_env.set_dr_training(True)
        render_eval = True

    # Optionally log success rate on a different DR percentage distribution
    custom_callbacks = None
    if args.other_eval_env_dr is not None:
        assert args.performance_lb is not None, 'The task-solved threshold must be defined when logging the success rate.'
        eval_search_space_id = args.eval_search_space_id if args.eval_search_space_id is not None else args.search_space_id
        custom_callbacks = [
                              get_succRateEvalCallback(dr_percentage=args.dr_percentage, search_space_id=args.search_space_id, performance_lb=args.performance_lb, prefix='custom_train_'),  # eval succ rate on training distr.
                              get_succRateEvalCallback(dr_percentage=args.other_eval_env_dr, search_space_id=eval_search_space_id, performance_lb=args.performance_lb, prefix='target_', best_model_save_path=run_path)         # eval succ rate on target distr.
                           ]


    print('--- Policy training start ---')
    mean_reward, std_reward, best_policy, which_one = policy.train(timesteps=args.timesteps,
                                                                   stopAtRewardThreshold=args.reward_threshold,
                                                                   n_eval_episodes=args.eval_episodes,
                                                                   eval_freq=args.eval_freq,
                                                                   best_model_save_path=run_path,
                                                                   return_best_model=True,
                                                                   verbose=args.verbose,
                                                                   eval_env=eval_env,
                                                                   custom_callbacks=custom_callbacks,
                                                                   render_eval=render_eval)
    env.set_dr_training(False)

    policy.save_state_dict(run_path+"final_model.pth")
    policy.save_full_state(run_path+"final_full_state.zip")
    print('--- Policy training done ----')

    print('\n\nMean reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["train_mean_reward"] = mean_reward
    wandb.run.summary["train_std_reward"] = std_reward
    wandb.run.summary["which_best_model"] = which_one

    torch.save(best_policy, run_path+"overall_best.pth")
    wandb.save(run_path+"overall_best.pth")

    # Free up some memory
    del env
    del custom_callbacks
    gc.collect()

    ### Evaluation on test environment
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args}, env_kwargs=env_kwargs)
    test_env.set_dr_distribution(dr_type='uniform', distr=training_bounds)
    test_env.set_dr_training(True)
    policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
    policy.load_state_dict(best_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes)
    print('Test reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["test_mean_reward"] = mean_reward
    wandb.run.summary["test_std_reward"] = std_reward



    if args.compute_final_univariate_eval:
        ### Evaluate return per dynamics parameter on global wide bounds (using 1 now)
        test_env = gym.make(args.test_env, **env_kwargs)
        test_env = make_wrapped_environment(test_env, args=args)
        policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
        policy.load_state_dict(best_policy)
        n_points_per_task_dim = 50 if not args.debug else 5
        test_episodes = 10 if not args.debug else 1
        return_per_dyn = np.empty((len(gt_task), n_points_per_task_dim))

        # evaluate on the total parameter range (95%)
        test_bounds = test_env.get_uniform_dr_by_percentage(percentage=0.95,
                                                            nominal_values=gt_task,
                                                            lower_bounds=lower_bounds)
        bounds_low, bounds_high = test_bounds[::2], test_bounds[1::2]

        for i in range(len(gt_task)):
            test_tasks = np.linspace(bounds_low[i], bounds_high[i], n_points_per_task_dim)
            curr_task = gt_task.copy()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
            for j, test_task in enumerate(test_tasks):
                curr_task[i] = test_task  # Only change one dimension at a time w.r.t. gt_task
                test_env.set_task(*curr_task)
                mean_reward, std_reward = policy.eval(n_eval_episodes=test_episodes)
                wandb.log({"avg_return_per_"+str(test_env.dyn_ind_to_name[i]): mean_reward, test_env.dyn_ind_to_name[i]: float(test_task)})
                return_per_dyn[i, j] = mean_reward

            wandb.run.summary[f"mean_return_full_range_dim{i}"] = np.mean(return_per_dyn[i,:])  # mean return on file 95% DR range
            wandb.run.summary[f"max_return_full_range_dim{i}"] = np.max(return_per_dyn[i,:])    # max return on file 95% DR range
            wandb.run.summary[f"min_return_full_range_dim{i}"] = np.min(return_per_dyn[i,:])    # max return on file 95% DR range
            wandb.run.summary[f"p98_return_full_range_dim{i}"] = np.quantile(return_per_dyn[i,:], q=.98)  # 98th-percentile return on file 95% DR range
            wandb.run.summary[f"p02_return_full_range_dim{i}"] = np.quantile(return_per_dyn[i,:], q=.02)  # 2nd-percentile return on file 95% DR range

            ax.plot(test_tasks, return_per_dyn[i, :], c='blue', linestyle='-')
            ax.set_ylim(min(0, np.min(return_per_dyn[i, :])), max(test_env.get_reward_threshold(), np.max(return_per_dyn[i, :])))  # Set a common scale for the y-axis
            ax.axvline(x=(training_bounds[i*2]), color='black', linestyle='--')
            ax.axvline(x=(training_bounds[i*2 + 1]), color='black', linestyle='--')
            ax.axvline(x=gt_task[i], color='red', linestyle='--')
            shade_start = training_bounds[i*2]
            shade_end = training_bounds[i*2 + 1]
            ax.axvspan(shade_start, shade_end, facecolor='grey', alpha=0.25)
            ax.set_xlabel(test_env.dyn_ind_to_name[i])
            ax.set_ylabel('Avg return')
            wandb.log({"avg_return_per_"+str(test_env.dyn_ind_to_name[i]): wandb.Image(fig)})
            plt.savefig(os.path.join(run_path, 'avg_return_per_'+test_env.dyn_ind_to_name[i]+'.png'))
            plt.close(fig)
        np.save(os.path.join(run_path, 'return_per_dyn.npy'), return_per_dyn)
        wandb.run.summary["mean_return_full_range"] = np.mean(return_per_dyn)
        wandb.run.summary["max_return_full_range"] = np.max(return_per_dyn)
        wandb.run.summary["min_return_full_range"] = np.min(return_per_dyn)
        wandb.run.summary["p98_return_full_range"] = np.quantile(return_per_dyn, q=.98)
        wandb.run.summary["p02_return_full_range"] = np.quantile(return_per_dyn, q=.02)

    wandb.finish()


def get_succRateEvalCallback(dr_percentage, search_space_id, performance_lb, prefix='target_', best_model_save_path=None, center_task=None):
    """Returns custom callback for logging
       success rate on a defined dr_percentage"""

    # Build target env
    curr_env_kwargs = {**env_kwargs, 'search_space_id': search_space_id} if len(env_kwargs) > 0 else {}
    gt_task = gym.make(args.env, **curr_env_kwargs).get_task()
    lower_bounds = np.zeros(len(gt_task)) if is_locomotion_env(args.env) else None  # use zeros as lower_bounds for locomotion envs params
    target_dr_bounds = gym.make(args.env, **curr_env_kwargs).get_uniform_dr_by_percentage(percentage=dr_percentage,
                                                                                                                             nominal_values=gt_task,
                                                                                                                             lower_bounds=lower_bounds,
                                                                                                                             dyn_mask=args.rand_only)
    custom_env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args, 'wrapper': 'returnTracker'}, env_kwargs=curr_env_kwargs)
    custom_env.set_dr_distribution(dr_type='uniform', distr=target_dr_bounds)
    custom_env.set_dr_training(True)
    custom_env.env_method('set_expose_episode_stats', **{'flag': True})  # track episode returns
    custom_env.env_method('reset_buffer')

    print(f'{prefix} env bounds:', target_dr_bounds)

    cb = EvalCallback(custom_env,
                 best_model_save_path=None,
                 eval_freq=args.eval_freq,
                 n_eval_episodes=args.eval_episodes,
                 deterministic=True,
                 callback_after_eval=WandbRecorderSuccessRateCallback(custom_env=custom_env, performance_lb=performance_lb, prefix=prefix, best_model_save_path=best_model_save_path),
                 verbose=args.verbose,
                 render=False)

    return cb

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback

class WandbRecorderSuccessRateCallback(BaseCallback):
    """
    A custom callback that allows to log eval reward after every evaluation
    on wandb. In contrast to the other wrapper, this wrapper only logs
    the return, and assumes the env has been wrapped through the custom
    ReturnTracker wrapper.
    
    Note: self.training_env did not correctly return the custom_env (likely because
    I should use self.eval_env). Therefore, I'm passing it as a class parameter
    """
    def __init__(self, custom_env, performance_lb, prefix='target_', best_model_save_path=None, verbose=0):
        super(WandbRecorderSuccessRateCallback, self).__init__(verbose)

        self.custom_env = custom_env
        self.performance_lb = performance_lb

        self.prefix = prefix
        self.verbose = verbose

        self.best_model_save_path = best_model_save_path
        self.best_policy_succ_rate = -1


    def _on_step(self) -> bool:
        """
        This method is called as a child callback of the `EventCallback`,
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        assert isinstance(self.custom_env, VecEnv)
        assert self.custom_env.has_attr('reset_buffer')
        assert np.all(self.custom_env.env_method('get_expose_episode_stats'))
        
        """
            Note: .reshape(-1) works instead of self._flatten because we use EvalCallback
            instead of policy.eval(), which could different values if eval_episodes % now != 0
        """
        returns = np.array(self.custom_env.env_method('get_buffer')).reshape(-1)  # retrieve tracked returns, and flatten values
        succ_metrics = np.array(self.custom_env.env_method('get_succ_metric_buffer')).reshape(-1)  # retrieve metric for measuring success, and flatten values

        if len(succ_metrics) == 0:
            # Use returns for measuring success
            success_rate = torch.tensor(returns >= self.performance_lb, dtype=torch.float32).mean()
        else:
            # Use custom metric for measuring success
            success_rate = torch.tensor(succ_metrics >= self.performance_lb, dtype=torch.float32).mean()

        current_timestep = self.num_timesteps  # this number is already multiplied by the number of parallel envs

        if self.best_model_save_path is not None:
            if success_rate > self.best_policy_succ_rate:
                self.best_policy_succ_rate = success_rate
                wandb.run.summary[f'best_ts'] = current_timestep
                wandb.run.summary[f'best_target_succ_rate'] = success_rate

                torch.save(self.model.policy.state_dict(), os.path.join(self.best_model_save_path, f'{self.prefix}best_succ_rate.pth'))

        wandb.log({f"{self.prefix}mean_reward": np.mean(returns), "timestep": current_timestep})
        wandb.log({f"{self.prefix}success_rate": success_rate, "timestep": current_timestep})
        if len(succ_metrics) > 0:
            wandb.log({f"{self.prefix}mean_succ_metric": np.mean(succ_metrics), "timestep": current_timestep})

        # Reset buffer after every evaluation
        self.custom_env.env_method('reset_buffer')

        return True

    def _flatten(self, multi_list):
        """Flatten a list of lists with potentially
        different lenghts into a 1D np array"""
        flat_list = [] 
        for single_list in multi_list:
            flat_list += single_list
        return np.array(flat_list, dtype=np.float64)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',            default=None, type=str, help='Train gym env')
    parser.add_argument('--test_env',       default=None, type=str, help='Test gym env')
    parser.add_argument('--group',          default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo',           default='ppo', type=str, help='RL Algo (ppo, lstmppo, sac)')
    parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=1, type=int, help='Number of cpus for env parallelization')
    parser.add_argument('--timesteps', '-t', default=1000, type=int, help='Global training timesteps (will be spread out across all parallel envs)')
    parser.add_argument('--reward_threshold', default=False, action='store_true', help='Stop at reward threshold')
    parser.add_argument('--eval_freq',      default=10000, type=int, help='Global timesteps frequency for training evaluations')
    parser.add_argument('--eval_episodes',  default=50, type=int, help='# episodes for training evaluations')
    parser.add_argument('--test_episodes',  default=100, type=int, help='# episodes for test evaluations')
    parser.add_argument('--seed',           default=0, type=int, help='Random seed')
    parser.add_argument('--device',         default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--notes',          default=None, type=str, help='Wandb notes')
    parser.add_argument('--wandb',          default='online', type=str, help='Wandb mode. [online, offline, disabled]')
    parser.add_argument('--verbose',        default=0, type=int, help='Verbose integer value [0, 1]')
    parser.add_argument('--render_eval',    default=False, action='store_true', help='Render evaluation episodes')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--debug',          default=False, action='store_true', help='Debug flag. Used to speed up some steps when they are just being tested.')

    parser.add_argument('--dr_percentage',  default=0.1, type=float, help='Percentage of ground truth values used to build the Uniform DR distribution. gt +- gt*percentage')
    parser.add_argument('--rand_only',      default=None, type=int, nargs='+', help='Index of dynamics parameter to randomize, instead of randomizing all possible parameters.')
    parser.add_argument('--rand_all_but',   default=None, type=int, help='Helper parameter that sets --rand_only [] to all indexes except for the one specified by --rand_all_but.')
    parser.add_argument('--dyn_in_obs',     default=False, action='store_true', help='If True, concatenate the dynamics of the environment in the observation vector, for task-aware policies.')
    parser.add_argument('--other_eval_env_dr', default=None, type=float, help='Test policy while training on a DR percentage different than the training one (e.g. max entropy 0.95 distribution)')
    parser.add_argument('--performance_lb',    default=None, type=float, help='Task-solved threshold in case other_eval_env_dr is set, in order to compute the success rate.')
    parser.add_argument('--compute_final_univariate_eval', default=False, action='store_true', help='If set, compute return over changing dynamics, separately for each randomized dimension and save these to a file.')

    # Params for asymmetric information
    parser.add_argument('--actor_state_only',   default=False, action='store_true', help='History or dynamics are filtered out from the actor input')
    parser.add_argument('--actor_history_only', default=False, action='store_true', help='Dynamics are filtered out from the actor input')
    parser.add_argument('--critic_dyn_only',    default=False, action='store_true', help='History is filtered out from the critic input')
    parser.add_argument('--stack_history',  default=None, type=int, help='Stack a number of previous (obs, actions) into the current obs vector. If > 1, it allows for implicit online systId, hence adaptive behavior.')

    # Panda gym specific parameters
    parser.add_argument('--qacc_factor', default=0.3, type=float, help='PandaGym envs kwarg')
    parser.add_argument('--control_penalty_coeff', default=1.0, type=float, help='PandaGym envs kwarg')
    parser.add_argument('--task_reward', default='target', type=str, help='PandaGym envs kwarg')
    parser.add_argument('--search_space_id', default=1., type=str, help='PandaGym envs kwarg')
    parser.add_argument('--eval_search_space_id', default=None, type=int, help='PandaGym env kwarg')
    # parser.add_argument('--center_task', default=None, type=float, help='Extra flexibility on top of dr_percentage. You can specify different nominal_values, instead of them being in the center of the search space')
    parser.add_argument('--center_task_id', default=None, type=int, help='Select a center task from predefined ones')

    return parser.parse_args()

args = parse_args()

# Get environment kwargs
env_kwargs = get_env_kwargs(args.env, args)
args.env_kwargs = env_kwargs

if __name__ == '__main__':
    main()