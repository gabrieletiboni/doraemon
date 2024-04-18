"""Train a policy with Automatic Domain Randomization
   (https://arxiv.org/abs/1910.07113)

    Examples:

        (DEBUG)
            python train_autodr.py --wandb disabled --env RandomContinuousInvertedCartPoleEasy-v0 -t 2000 --eval_episodes 1 --test_episodes 1 --seed 42 --dr_percentage 0.2 --algo sac --delta 0.1 --buffer_size 1 --check_update_freq 500 --verbose 2 --debug
        
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
from autodr.autodr import TrainingSubRtn, UniformDistribution, BetaDistribution, AutoDR

def main():
    # args.eval_freq = max(args.eval_freq // args.now, 1)   # Making eval_freq behave w.r.t. global timesteps, so it follows --timesteps convention
    torch.set_num_threads(max(5, args.now))  # hard-coded for now. Avoids taking up all CPUs when parallelizing with multiple environments and processes on hephaestus

    assert args.dr_percentage <= 1 and args.dr_percentage >= 0
    assert args.env is not None
    assert args.test_env is None, 'source and target domains should be the same. As of right now, test_env is used to test the policy on the final target DR distribution'
    if args.test_env is None:
        args.test_env = args.env

    assert args.threshold_low < args.threshold_high
    assert args.bound_sampling_prob > 0 and args.bound_sampling_prob < 1
    assert args.delta > 0. and args.delta <= 0.5, 'Update step for each dimension should be below 50\% of the whole maximum width the distribution can reach.'
    assert args.check_update_freq / args.now >= gym.make(args.env, **env_kwargs)._max_episode_steps, f'The subroutine is stopped before each worker can reach the max episode timesteps ({gym.make(args.env, **env_kwargs)._max_episode_steps}), potentially preventing montecarlo returns to be saved. Is this desired?'


    init_task = gym.make(args.env, **env_kwargs).get_task()  # initial dynamics parameters (static vector)
    if args.rand_all_but is not None:  # args.rand_all_but overwrites args.rand_only
        args.rand_only = np.arange(len(init_task)).tolist()
        del args.rand_only[args.rand_all_but]


    ### Configs and Wandb
    random_string = get_random_string(5)
    run_name = "AutoDR_"+ args.algo +'_seed'+str(args.seed)+'_'+random_string
    print(f'========== RUN_NAME: {run_name} ==========')
    pprint(vars(args))
    set_seed(args.seed)
    wandb.init(config=vars(args),
               project="DORAEMON-dev",
               group="AutoDR_"+str(args.env if args.group is None else args.group),
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


    ### Get init and bounds for AutoDR
    print('Ground truth task:', init_task)
    lower_bounds = np.zeros(len(init_task)) if is_locomotion_env(args.env) else None  # use zeros as lower_bounds for locomotion envs params
    bounds = gym.make(args.env, **env_kwargs).get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                                      nominal_values=init_task,
                                                                      lower_bounds=lower_bounds,
                                                                      dyn_mask=args.rand_only)
    print('Maximum bounds:', bounds)
    bounds_low, bounds_high = bounds[::2], bounds[1::2]
    init_distr = []
    uniform_bounds = []

    # Get starting point in dynamics space
    if args.start_from_id is not None:
        # center in space `start_from_id`
        assert 'DMPandaPush-' in args.env, 'This function is for PandaPush envs only for now.'
        assert args.original, 'start_from_id without original has not been implemented yet.' \
                              'To do it, you also need to change the get_init_distr_width_percentage' \
                              'function such that the entropy is the same and also such that you dont' \
                              'set the initial bounds to be outside of the boundaries'
        starting_task = get_starting_task(start_from_id=args.start_from_id,
                                          args=args)
    else:
        # center of target search space `search_space_id`
        starting_task = init_task

    # Get initial uniform width such that initial entropy is the same as doraemon (if not args.original)
    init_distr_width_percentage = get_init_distr_width_percentage(desired_entropy=get_init_entropy(args.env), bounds=list(zip(bounds_low, bounds_high)))

    for i, (m, M) in enumerate(zip(bounds_low, bounds_high)):
        if args.original:
            init_distr.append({'m': starting_task[i]-1e-5, 'M': starting_task[i]+1e-5})
        else:
            init_distr.append({'m': starting_task[i]-(M-m)*init_distr_width_percentage/2, 'M': starting_task[i]+(M-m)*init_distr_width_percentage/2})
        uniform_bounds.append({'m': m, 'M': M})
        assert init_distr[-1]['m'] > m and init_distr[-1]['M'] < M, 'The initial distribution cannot be set beyond the boundaries.'

    init_distribution = UniformDistribution(distr=init_distr)
    uniform_bounds = UniformDistribution(distr=uniform_bounds)
    print('init distr:')
    init_distribution.print()
    print('target distr:')
    uniform_bounds.print()


    ### Actor & Critic input observation masks for asymmetric information
    actor_obs_mask, critic_obs_mask = get_actor_critic_obs_masks(args)


    ### Set up training
    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args, 'wrapper': 'autodr'}, env_kwargs=env_kwargs)
    eff_lr = get_learning_rate(args, env)
    # eval_freq is not used for AutoDR
    # eval_freq = min(int(args.check_update_freq/args.now/2), args.eval_freq)  # make sure you at least evaluate the policy 2 times per iteration

    # Evaluation transitions are not used for AutoDR (boundary sampling, etc.)
    eval_env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args, 'wrapper': 'returnTracker'}, env_kwargs=env_kwargs)

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


    ### Launch AutoDR
    autoDR = AutoDR(training_subrtn=training_subrtn,
                    budget=args.timesteps,
                    init_distr=init_distribution,
                    boundaries=uniform_bounds,
                    performance_threshold_low=args.threshold_low,
                    performance_threshold_high=args.threshold_high,
                    check_update_freq=args.check_update_freq,
                    delta=args.delta,
                    buffer_size=args.buffer_size,
                    test_episodes=(args.test_episodes if not args.debug else 1),
                    train_until=args.train_until_lb,
                    force_success_with_returns=args.force_success_with_returns,
                    verbose=args.verbose)

    autoDR.learn(ckpt_dir=run_path)
    final_policy = autoDR.final_policy

    n_iters = len(autoDR.distr_history)
    print('Number of iterations:', n_iters)


    ### Save distributions to disk
    distr_dir = os.path.join(run_path, 'distributions')
    create_dir(distr_dir)
    save_object(autoDR.distr_history, save_dir=distr_dir, filename='distr_history')


    ### Free up some memory
    del training_subrtn
    del autoDR
    del env
    gc.collect()


    ### Evaluation on target environment
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv, wrapper_class=make_wrapped_environment, wrapper_kwargs={'args': args}, env_kwargs=env_kwargs)
    test_env.set_dr_distribution(dr_type='uniform', distr=uniform_bounds.get_params())
    test_env.set_dr_training(True)
    policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
    policy.load_state_dict(final_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes)
    print(f'Test reward: {mean_reward} +- {std_reward}')

    wandb.run.summary["test_mean_reward"] = mean_reward
    wandb.run.summary["test_std_reward"] = std_reward


    ### Compute joint 2D heatmap values
    del test_env
    if args.compute_final_heatmap:
        print('\n--- Computing joint 2D heatmap values')
        compute_joint_2dheatmap_data(final_policy, run_path)

    wandb.finish()


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

def get_init_distr_width_percentage(desired_entropy, bounds):
    """Return the initial uniform width in % for each dimension around
    the initial value such that the entropy is = desired_entropy
    """
    # solve for L: desired_entropy = sum log( (M[i] - m[i])*L )

    sum_log = 0
    n = 0
    for i, (m, M) in enumerate(bounds):
        # init_distr.append({'m': init_task[i]-1e-5, 'M': init_task[i]+1e-5})
        # uniform_bounds.append({'m': m, 'M': M})
        sum_log += np.log(M-m)
        n += 1

    L = np.exp( (desired_entropy - sum_log) / n )
    return L


def get_starting_task(start_from_id, args):
    """Get initial location if start_from_id is specified.
        This function is for PandaPush envs only for now.
    """
    assert 'DMPandaPush-' in args.env, 'This function is for PandaPush envs only for now.'

    starting_env = gym.make(args.env, **{**args.env_kwargs, 'search_space_id': start_from_id})  # env with starting space
    starting_task = starting_env.get_task()

    return starting_task


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

    init_task = gym.make(args.test_env, **env_kwargs).get_task()
    lower_bounds = np.zeros(len(init_task)) if is_locomotion_env(args.test_env) else None  # use zeros as lower_bounds for locomotion envs params
    test_bounds = gym.make(args.test_env, **env_kwargs).get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                                       nominal_values=init_task,
                                                                       lower_bounds=lower_bounds)
    
    bounds_low, bounds_high = test_bounds[::2], test_bounds[1::2]

    test_tasks_1 = np.linspace(bounds_low[dyn_pair[0]], bounds_high[dyn_pair[0]], n_points_per_task_dim) # (50,)
    test_tasks_2 = np.linspace(bounds_low[dyn_pair[1]], bounds_high[dyn_pair[1]], n_points_per_task_dim) # (50,)

    curr_task = init_task.copy()
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




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', '-t',default=1000, type=int, help='Budget. Global training timesteps (will be spread out across all parallel envs)')
    parser.add_argument('--env',            default=None, type=str, help='Train gym env')
    parser.add_argument('--test_env',       default=None, type=str, help='Test gym env')
    parser.add_argument('--group',          default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo',           default='sac', type=str, help='RL Algo (ppo, lstmppo, sac)')
    parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=1, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--eval_freq',      default=10000, type=int, help='Global timesteps frequency for training evaluations')
    parser.add_argument('--eval_episodes',  default=50, type=int, help='# episodes for training evaluations')
    parser.add_argument('--test_episodes',  default=100, type=int, help='# episodes for test evaluations')
    parser.add_argument('--seed',           default=0, type=int, help='Random seed')
    parser.add_argument('--device',         default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--notes',          default=None, type=str, help='Wandb notes')
    parser.add_argument('--wandb',          default='online', type=str, help='Wandb mode. [online, offline, disabled]')
    parser.add_argument('--verbose',        default=1, type=int, help='Verbose integer value')
    parser.add_argument('--stack_history',  default=None, type=int, help='Stack a number of previous (obs, actions) into the current obs vector. If > 1, it allows for implicit online systId, hence adaptive behavior.')
    parser.add_argument('--dr_percentage',  default=0.1, type=float, help='Percentage of values used to build the DR distribution bounds. mean +- mean*percentage')
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

    # AutoDR-specific params
    parser.add_argument('--original',            default=False, action='store_true', help='Original implementation: collapsed initial distribution')
    parser.add_argument('--delta',               default=0.1, type=float, help='Relative Update step size normalized in [0, 1]')
    parser.add_argument('--threshold_low',       default=0., type=float, help='Performance threshold low')
    parser.add_argument('--threshold_high',      default=1., type=float, help='Performance threshold high')
    parser.add_argument('--bound_sampling_prob', default=0.5, type=float, help='Boundary sampling probability')
    parser.add_argument('--buffer_size',         default=240, type=int, help='Performance data buffer size')
    parser.add_argument('--check_update_freq',   default=100, type=int, help='Check performance buffers frequency in timesteps')
    parser.add_argument('--train_until_lb',      default=False, action='store_true', help='Train on initial distribution until performance lower bound is reached')
    parser.add_argument('--force_success_with_returns', default=False, action='store_true', help='If set, force using returns as to measure average performance even if env.success_metric is defined. Proper corresponding thresholds need to be defined.')

    # Panda gym specific parameters
    parser.add_argument('--qacc_factor', default=0.3, type=float, help='PandaGym envs kwarg')
    parser.add_argument('--control_penalty_coeff', default=1.0, type=float, help='PandaGym envs kwarg')
    parser.add_argument('--task_reward', default='target', type=str, help='PandaGym envs kwarg')
    parser.add_argument('--search_space_id', default=1, type=str, help='PandaGym envs kwarg')
    parser.add_argument('--start_from_id', default=None, type=int, help='PandaGym envs kwarg (start at center of this space instead of at the center of search_space_id)')
    parser.add_argument('--absolute_acc_pen', default=False, action='store_true', help='PandaGym envs kwarg')

    return parser.parse_args()

args = parse_args()

# Get environment kwargs
env_kwargs = get_env_kwargs(args.env, args)
args.env_kwargs = env_kwargs

if __name__ == '__main__':
    main()