"""Train a policy with LSDR
   (https://ieeexplore.ieee.org/document/9341019)

    Examples:

        (DEBUG)
            python train_lsdr.py --wandb disabled --env RandomContinuousInvertedCartPoleEasy-v0 --gradient_steps 1 --eval_episodes 1 --test_episodes 1 --seed 42 --dr_percentage 0.2 --algo sac --performance_lb 0 --verbose 2 --debug --timesteps_per_iter 500 --timesteps 1500 --count_eval_ts_in_budget --whiten_performance --n_contexts_for_eval 5 --distr_learning_iters 10 --use_kl_regularizer
            python train_lsdr.py --wandb disabled --env DMPandaPush-FFPosCtrl-ContPen-GoalA-mf0comy_d_fl-JVelNoise0.0011-BoxNoise0.002-InitBox0.01-BoxHeight0.005-NormReward-v0 --gradient_steps 1 --eval_episodes 1 --test_episodes 1 --seed 42 --dr_percentage 1.0 --algo sac --performance_lb 0 --verbose 2 --debug --timesteps_per_iter 500 --timesteps 1500 --count_eval_ts_in_budget --whiten_performance --n_contexts_for_eval 5 --distr_learning_iters 10 --use_kl_regularizer --search_space_id 1 --start_from_id 2 --qacc_factor 0.9

        (See readme.md for reproducing paper results)
"""
from pprint import pprint
import argparse
import pdb
import sys
import socket
import os
import pickle
import gc

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
from lsdr.lsdr import TrainingSubRtn, UniformDistribution, MultivariateGaussian, LSDR
from autodr.autodr import BetaDistribution

def main():
    torch.set_num_threads(max(5, args.now))  # hard-coded for now. Avoids taking up all CPUs when parallelizing with multiple environments and processes on hephaestus

    assert args.dr_percentage <= 1 and args.dr_percentage >= 0
    assert args.env is not None
    assert args.test_env is None, 'source and target domains should be the same. As of right now, test_env is used to test the policy on the final target DR distribution'
    if args.test_env is None:
        args.test_env = args.env


    gt_task = gym.make(args.env,**env_kwargs).get_task()  # ground truth dynamics parameters (static vector)
    if args.rand_all_but is not None:  # args.rand_all_but overwrites args.rand_only
        args.rand_only = np.arange(len(gt_task)).tolist()
        del args.rand_only[args.rand_all_but]


    ### Configs and Wandb
    random_string = get_random_string(5)
    run_name = "LSDR_"+ args.algo +'_seed'+str(args.seed)+'_'+random_string
    print(f'========== RUN_NAME: {run_name} ==========')
    pprint(vars(args))
    set_seed(args.seed)
    wandb.init(config=vars(args),
               project="DORAEMON-dev",
               group="LSDRv3_"+str(args.env if args.group is None else args.group),
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

    ### Get init and target distributions for LSDR
    print('Ground truth task:', gt_task)
    lower_bounds = np.zeros(len(gt_task)) if is_locomotion_env(args.env) else None  # use zeros as lower_bounds for locomotion envs params
    target_training_bounds = gym.make(args.env, **env_kwargs).get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                                      nominal_values=gt_task,
                                                                      lower_bounds=lower_bounds,
                                                                      dyn_mask=args.rand_only)
    print('Target training bounds:', target_training_bounds)
    bounds_low, bounds_high = target_training_bounds[::2], target_training_bounds[1::2]

    # Get starting point in dynamics space
    if args.start_from_id is not None:
        # center in space `start_from_id`
        assert 'DMPandaPush-' in args.env, 'This function is for PandaPush envs only for now.'
        init_mean = get_starting_task(start_from_id=args.start_from_id,
                                      low=bounds_low,
                                      high=bounds_high,
                                      args=args)
    else:
        # center of target search space `search_space_id`
        # 0.5 as the gaussian is standardized in [0, 1] for opt. purposes
        init_mean = np.repeat(0.5, len(gt_task))

    # Get initial variance such that init entropy is = to init DORAEMON entropy
    init_variance = get_init_variance(desired_entropy=get_init_entropy(args.env), low=bounds_low, high=bounds_high, ndims=len(gt_task), start_from_id=args.start_from_id)
    init_distr_means = []
    init_distr_variances = []
    init_distr = {'mean': None, 'cov': None, 'low': [], 'high': []}  # low and high for denormalizing parameters
    uniform_target = []  # Uniform distr
    for i, (m, M) in enumerate(zip(bounds_low, bounds_high)):
        init_distr_means.append( init_mean[i] )  # (M + m) / 2
        init_distr_variances.append(init_variance)
        init_distr['low'].append(m)
        init_distr['high'].append(M)
        uniform_target.append({'m': m, 'M': M})


    init_distr['mean'] = np.array(init_distr_means)
    init_distr['cov'] = np.diag(init_distr_variances)
    init_distr['low'] = np.array(init_distr['low'])
    init_distr['high'] = np.array(init_distr['high'])
    init_distribution = MultivariateGaussian(distr=init_distr, now=args.now)

    target_distribution = UniformDistribution(distr=uniform_target)

    print('TEMPORARY SANITY CHECK: Des init:', get_init_entropy(args.env) , 'Actual init entropy:', init_distribution.entropy().detach().item())

    print('init distr:')
    init_distribution.print()
    print('target distr:')
    target_distribution.print()


    ### Actor & Critic input observation masks for asymmetric information
    actor_obs_mask, critic_obs_mask = get_actor_critic_obs_masks(args)


    ### Set up training
    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args, 'wrapper': 'lsdr'}, env_kwargs=env_kwargs)
    eval_env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args, 'wrapper': 'returnTracker'}, env_kwargs=env_kwargs)
    eff_lr = get_learning_rate(args, env)

    assert args.timesteps_per_iter//args.now >= gym.make(args.env, **env_kwargs)._max_episode_steps, 'Error: training timesteps of {args.timesteps_per_iter} do not allow to collect a full episode.'

    training_subrtn = TrainingSubRtn(env,
                                     eval_env=eval_env,
                                     algo=args.algo,
                                     lr=eff_lr,
                                     gamma=args.gamma,
                                     device=args.device,
                                     seed=args.seed,
                                     actor_obs_mask=actor_obs_mask,
                                     critic_obs_mask=critic_obs_mask,
                                     n_eval_episodes=args.eval_episodes,
                                     run_path=run_path,
                                     gradient_steps=args.gradient_steps,
                                     verbose=args.verbose)



    ### Launch LSDR
    lsdr = LSDR(training_subrtn=training_subrtn,
                performance_lower_bound=args.performance_lb,
                init_distr=init_distribution,
                target_distr=target_distribution,
                budget=args.timesteps,
                training_steps=args.timesteps_per_iter,
                distr_learning_iters=args.distr_learning_iters,
                test_episodes=(args.test_episodes if not args.debug else 1),
                force_success_with_returns=args.force_success_with_returns,
                count_eval_ts_in_budget=args.count_eval_ts_in_budget,
                n_contexts_for_eval=args.n_contexts_for_eval,
                alpha=args.alpha,
                whiten_performance=args.whiten_performance,
                standardized_performance=args.standardized_performance,
                baseline=args.baseline,
                use_kl_regularizer=args.use_kl_regularizer,
                obj_fun_lr=args.obj_fun_lr,
                training_subrtn_kwargs={},
                verbose=args.verbose)

    lsdr.learn(ckpt_dir=run_path)

    # best_policy = lsdr.best_policy  # best performance on target distribution
    # torch.save(lsdr.best_policy, os.path.join(run_path, 'best_on_target.pth'))
    last_policy = lsdr.final_policy  # policy at last iteration
    n_iters = len(lsdr.distr_history)


    ### Plot distributions
    # fig, ax = plt.subplots(nrows=1, ncols=init_distribution.ndims, figsize=(8,5))
    # alpha_step = 1/(eff_n_iters+1)
    # for i, distr in enumerate(lsdr.distr_history):
    #     if i == len(lsdr.distr_history) - 1:  # last distribution
    #         target_distribution.visualize_distr(ax, alpha=0.9, color='red', label='Target')
    #         distr.visualize_distr(ax, alpha=0.9, color='#FFEB3B', label='Last')
    #         lsdr.best_distr.visualize_distr(ax, alpha=0.9, color='#43A047', label='Best')
    #     else:
    #         distr.visualize_distr(ax, alpha=(int(i+1)*alpha_step), color='blue', label=None)
    # plt.legend()
    # wandb.log({"distr_history": wandb.Image(fig)})
    # plt.savefig(os.path.join(run_path, 'lsdr_distr_history.png'))
    # plt.close()


    ### Save distributions to disk
    distr_dir = os.path.join(run_path, 'distributions')
    create_dir(distr_dir)
    save_object(lsdr.distr_history, save_dir=distr_dir, filename='distr_history')
    save_object(lsdr.best_distr, save_dir=distr_dir, filename='best_distr')


    ### Free up some memory
    del training_subrtn
    del lsdr
    del env
    gc.collect()


    ### Evaluation on target environment
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args}, env_kwargs=env_kwargs)
    test_env.set_dr_distribution(dr_type='uniform', distr=target_training_bounds)
    test_env.set_dr_training(True)
    policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
    policy.load_state_dict(last_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes)
    print(f'{"-"*50}\nTest reward and stdev: {mean_reward} +/- {std_reward}')

    wandb.run.summary["test_mean_reward"] = mean_reward
    wandb.run.summary["test_std_reward"] = std_reward


    ### Compute joint 2D heatmap values
    del test_env
    if args.compute_final_heatmap:
        print('\n--- Computing joint 2D heatmap values')
        compute_joint_2dheatmap_data(last_policy, run_path)

    wandb.finish()



def compute_joint_2dheatmap_data(test_policy, run_path):
    """Compute data for joint 2d-heatmap visualization"""
    dyn_pair = list(get_dyn_pair_indexes_per_env(args.test_env))

    save_dir = os.path.join(run_path, 'joint_avg_return_per_dyn')
    create_dirs(save_dir)
    target_filename = os.path.join(save_dir, f'joint_return_per_dyns_{dyn_pair[0]}_{dyn_pair[1]}.npy')

    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args}, env_kwargs=env_kwargs)

    actor_obs_mask, critic_obs_mask = get_actor_critic_obs_masks(args)
    policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
    policy.load_state_dict(test_policy)

    n_points_per_task_dim = 50 if not args.debug else 5
    return_per_dyn = np.empty((n_points_per_task_dim, n_points_per_task_dim))

    gt_task = gym.make(args.test_env, **env_kwargs).get_task()
    lower_bounds = np.zeros(len(gt_task)) if is_locomotion_env(args.test_env) else None  # use zeros as lower_bounds for locomotion envs params
    test_bounds = gym.make(args.test_env, **env_kwargs).get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                                       nominal_values=gt_task,
                                                                       lower_bounds=lower_bounds)
    
    bounds_low, bounds_high = test_bounds[::2], test_bounds[1::2]

    test_tasks_1 = np.linspace(bounds_low[dyn_pair[0]], bounds_high[dyn_pair[0]], n_points_per_task_dim) # (50,)
    test_tasks_2 = np.linspace(bounds_low[dyn_pair[1]], bounds_high[dyn_pair[1]], n_points_per_task_dim) # (50,)

    curr_task = gt_task.copy()
    for j, test_task_1 in enumerate(test_tasks_1):
        for k, test_task_2 in enumerate(test_tasks_2):
            curr_task[dyn_pair] = [test_task_1, test_task_2]  # Change two params at a time, and keep others to the nominal values
            repeated_curr_task = np.repeat(curr_task[np.newaxis, :], args.now, axis=0)  # duplicate task args.now times to handle vec envs
            test_env.set_task(repeated_curr_task)
            mean_reward, std_reward = policy.eval(n_eval_episodes=(10 if not args.debug else 1))
            return_per_dyn[j, k] = mean_reward

            # Show progress
            print(f'[{j+1}/{n_points_per_task_dim}, {k+1}/{n_points_per_task_dim}]: {round(mean_reward, 2)} +- {round(std_reward,2)}', end="\r")

    # Print a new line after the loop finishes
    print()

    # Create dir and save matrix
    np.save(target_filename, return_per_dyn)


def get_init_entropy(env):
    """Return initial entropy as in DORAEMON with a beta distribution"""
    gt_task = gym.make(args.env, **env_kwargs).get_task()
    lower_bounds = np.zeros(len(gt_task)) if is_locomotion_env(env) else None  # use zeros as lower_bounds for locomotion envs params
    target_training_bounds = gym.make(env, **env_kwargs).get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                                        nominal_values=gt_task,
                                                                        lower_bounds=lower_bounds,
                                                                        dyn_mask=args.rand_only)
    bounds_low, bounds_high = target_training_bounds[::2], target_training_bounds[1::2]
    a_start, b_start = 100, 100 
    a_target, b_target = 1, 1
    init_distr = []
    target_distr = []
    for m, M in zip(bounds_low, bounds_high):
        init_distr.append({'m': m, 'M': M, 'a': a_start, 'b': b_start})
        target_distr.append({'m': m, 'M': M, 'a': a_target, 'b': b_target})


    init_doraemon_distribution = BetaDistribution(distr=init_distr)
    return init_doraemon_distribution.entropy().item()

def get_init_variance(desired_entropy, low, high, ndims, start_from_id):
    """Returns the corresponding initial variance of
    the multivariate normal distribution given the desired
    initial entropy.

    The gaussian is initialized to an independent gaussian with
    equal diagonal variances, as each dimension is considered
    normalized in [0, 1] as the truncated search space.
    """

    # Explicit formula using non-truncated gaussian entropy
    # (good approximation for the narrow starting distribution)
    # the np.sum(np.log(np.array(bounds_high) - np.array(bounds_low))) term
    # accounts for the normalized space, as Y = AX + b => H(Y) = H(X) + log(|det A|)
    init_variance = np.exp( 2*(desired_entropy - np.sum(np.log(np.array(high) - np.array(low))) - ndims/2 - np.log(2*np.pi)*ndims/2)/ndims )

    if 'DMPandaPush-' in args.env:
        if start_from_id is not None:
            # Manually adjust the variance of the starting gaussian to match
            # the same initial variance as Doraemon
            if start_from_id == 2:
                # Des init entropy: -22.7283 vs. Actual init entropy: -22.7277
                init_variance *= 2.226

    return init_variance


def get_starting_task(start_from_id, low, high, args):
    """Get initial location if start_from_id is specified.
        This function is for PandaPush envs only for now.
    """
    assert 'DMPandaPush-' in args.env, 'This function is for PandaPush envs only for now.'

    starting_env = gym.make(args.env, **{**args.env_kwargs, 'search_space_id': start_from_id})  # env with starting space
    starting_task = starting_env.get_task()

    # Normalize task in [0, 1]
    norm_starting_task = (starting_task - low) / (high - low)

    return norm_starting_task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps','-t', default=1000, type=int, help='Budget. Global environment timesteps used (will be spread out across all parallel envs and monte-carlo returns for objective function evaluation)')
    parser.add_argument('--env',            default=None, type=str, help='Train gym env')
    parser.add_argument('--test_env',       default=None, type=str, help='Test gym env')
    parser.add_argument('--group',          default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo',           default='sac', type=str, help='RL Algo (ppo, lstmppo, sac)')
    parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=1, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--eval_episodes',  default=50, type=int, help='# episodes for training evaluations')
    parser.add_argument('--test_episodes',  default=100, type=int, help='# episodes for test evaluations')
    parser.add_argument('--seed',           default=0, type=int, help='Random seed')
    parser.add_argument('--device',         default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--notes',          default=None, type=str, help='Wandb notes')
    parser.add_argument('--wandb',          default='online', type=str, help='Wandb mode. [online, offline, disabled]')
    parser.add_argument('--verbose',        default=1, type=int, help='Verbose integer value')
    parser.add_argument('--stack_history',  default=None, type=int, help='Stack a number of previous (obs, actions) into the current obs vector. If > 1, it allows for implicit online systId, hence adaptive behavior.')
    parser.add_argument('--dr_percentage',  default=0.1, type=float, help='Percentage of ground truth values used to build the Uniform DR distribution. gt +- gt*percentage')
    parser.add_argument('--rand_only',      default=None, type=int, nargs='+', help='Index of dynamics parameter to randomize, instead of randomizing all possible parameters.')
    parser.add_argument('--rand_all_but',   default=None, type=int, help='Helper parameter that sets --rand_only [] to all indexes except for the one specified by --rand_all_but.')
    parser.add_argument('--dyn_in_obs',     default=False, action='store_true', help='If True, concatenate the dynamics of the environment in the observation vector, for task-aware policies.')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Number of gradient steps when policy is updated in sb3 using SAC. -1 means as many as --args.now')
    parser.add_argument('--debug',          default=False, action='store_true', help='Debug flag. Used to speed up some steps when they are just being tested.')
    parser.add_argument('--compute_final_heatmap', default=False, action='store_true', help='If set, compute 2D heatmap at the end of training and save results to file.')

    # Params for asymmetric information
    parser.add_argument('--actor_state_only',   default=False, action='store_true', help='History or dynamics are filtered out from the actor input')
    parser.add_argument('--actor_history_only', default=False, action='store_true', help='Dynamics are filtered out from the actor input')
    parser.add_argument('--critic_dyn_only',    default=False, action='store_true', help='History is filtered out from the critic input')

    # LSDR params
    parser.add_argument('--timesteps_per_iter', default=100, type=int, help='Training env steps between each LSDR distribution update')
    parser.add_argument('--performance_lb',  default=None, type=float, help='Used as threshold for computing success rate', required=True)
    parser.add_argument('--distr_learning_iters', default=10, type=int, help='Gradient descent iterations at each distribution update.')
    parser.add_argument('--alpha', default=1.0, type=float, help='trade-off parameter in opt. problem')
    parser.add_argument('--count_eval_ts_in_budget', default=False, action='store_true', help='Count monte carlo timesteps for obj. function evaluation towards the total budget available.')
    parser.add_argument('--whiten_performance', default=False, action='store_true', help='standardize returns for a consistent objective function scale')
    parser.add_argument('--standardized_performance', default=False, action='store_true', help='Standardize returns with in-batch statistics directly.')
    parser.add_argument('--force_success_with_returns', default=False, action='store_true', help='If set, force using returns as a success metric condition even if env.success_metric is defined. A proper corresponding performance_lb needs to be defined.')
    parser.add_argument('--use_kl_regularizer', default=False, action='store_true', help='compute KL divergence empirically, assuming independence')
    parser.add_argument('--obj_fun_lr', default=1e-3, type=float, help='Learning rate of Adam optimizer for obj function gradient descent steps.')
    parser.add_argument('--n_contexts_for_eval', default=100, type=int, help='Number of dynamics parameter samples for MC evaluation')

    # Panda gym specific parameters
    parser.add_argument('--qacc_factor', default=0.3, type=float, help='PandaGym envs kwarg')
    parser.add_argument('--control_penalty_coeff', default=1.0, type=float, help='PandaGym envs kwarg')
    parser.add_argument('--task_reward', default='target', type=str, help='PandaGym envs kwarg')
    parser.add_argument('--search_space_id', default=1, type=str, help='PandaGym envs kwarg')
    parser.add_argument('--start_from_id', default=None, type=int, help='PandaGym envs kwarg (start from this space instead of at the center of search_space_id)')
    parser.add_argument('--absolute_acc_pen', default=False, action='store_true', help='PandaGym envs kwarg')
    parser.add_argument('--baseline', default=None, type=float, help='PandaGym baseline for rewards')

    return parser.parse_args()

args = parse_args()

# Get environment kwargs
env_kwargs = get_env_kwargs(args.env, args)
args.env_kwargs = env_kwargs

if __name__ == '__main__':
    main()