"""Test a policy

    Example:
        python test_policy.py --run <run_path> --test_episodes 10 --render
"""
from pprint import pprint
import argparse
import pdb
import sys
from dataclasses import dataclass

import numpy as np
import gym
import dr_envs
from stable_baselines3.common.evaluation import evaluate_policy

from utils.utils import *
from utils.gym_utils import *
from policy.policy import Policy

@dataclass
class DictAsArgs:
    """Allows to access dict keys by attribute names"""
    config: dict
    def __getattr__(self, attr):
        return self.config.get(attr, None)

def main():
    pprint(vars(args))

    assert args.run is not None
    assert os.path.isdir(args.run)

    config = load_config(args.run)

    # Compatibility with Fixed-DR and PandaPush
    if 'eval_search_space_id' in config:
        print(f'NOTE! This run was trained on search space {config["search_space_id"]} but it\'s now being tested on DR{config["eval_search_space_id"]}')
        config['search_space_id'] = config['eval_search_space_id']  # replace training setting with test setting

        assert 'search_space_id' in config['env_kwargs']
        config.env_kwargs['search_space_id'] = config['eval_search_space_id']

    # Compatibility with UDR when training with Dr 0.0
    if 'other_eval_env_dr' in config:
        print(f'NOTE! This run was trained on DR{config["dr_percentage"]} but it\'s now being tested on DR{config["other_eval_env_dr"]}')
        config["dr_percentage"] = config["other_eval_env_dr"]

    if 'env_kwargs' not in config:
        config['env_kwargs'] = {}


    config = DictAsArgs(config)
    pprint(vars(config))


    ### Get test bounds to be the same as training bounds
    gt_task = gym.make(config.env, **config.env_kwargs).get_task()
    print('Nominal task:', gt_task)
    lower_bounds = np.zeros(len(gt_task)) if is_locomotion_env(config.env) else None  # use zeros as lower_bounds for locomotion envs params
    training_bounds = gym.make(config.env, **config.env_kwargs).get_uniform_dr_by_percentage(percentage=config.dr_percentage,
                                                                                             nominal_values=gt_task,
                                                                                             lower_bounds=lower_bounds,
                                                                                             dyn_mask=config.rand_only)
    print('Test bounds:', training_bounds)


    ### Actor & Critic input observation masks for asymmetric information
    actor_obs_mask, critic_obs_mask = get_actor_critic_obs_masks(config)
    print('actor obs mask:', actor_obs_mask)
    print('critic obs mask:', critic_obs_mask)
    

    ### Create env and load policy
    env = gym.make(config.env, **config.env_kwargs)
    env = make_wrapped_environment(env, args=config)
    env.set_dr_distribution(dr_type='uniform', distr=training_bounds)
    env.set_dr_training(True)

    if args.best_on_target:
        assert os.path.isfile(os.path.join(args.run, 'best_on_target.pth')) or os.path.isfile(os.path.join(args.run, 'target_best_succ_rate.pth'))
        policy_filename = 'best_on_target.pth' if os.path.isfile(os.path.join(args.run, 'best_on_target.pth')) else 'target_best_succ_rate.pth'
        policy = Policy(algo=config.algo, env=env, device=config.device, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
        policy.load_state_dict(os.path.join(args.run, policy_filename))
    else:
        # Fall back to overall_best.pth or best_model.zip (careful that these mean different things for UDR, DORAEMON ...)
        if os.path.isfile(os.path.join(args.run, 'overall_best.pth')):
            policy = Policy(algo=config.algo, env=env, device=config.device, actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
            policy.load_state_dict(os.path.join(args.run, 'overall_best.pth'))

        elif os.path.isfile(os.path.join(args.run, 'best_model.zip')):
            print('WARNING! overall_best.pth model not found, falling back to current best_model.zip (the run likely has not finished training)')
            policy = Policy(algo=config.algo, env=env, device=config.device, load_from_pathname=os.path.join(args.run, 'best_model.zip'), actor_obs_mask=actor_obs_mask, critic_obs_mask=critic_obs_mask)
        else:
            raise ValueError(f'No model has been found in current run path: {args.run}')


    print('============================')
    print('Env:', config.env)
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Task dim:', env.task_dim)
    print('Nominal values:', env.nominal_values)
    
    ### Implicit eval loop
    # assert not args.discounted, 'Cannot compute discounted return with sb3'
    # mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes, render=args.render)
    # print('Mean reward:', mean_reward)
    # print('Std reward:', std_reward)

    ### Explicit eval loop
    obs = env.reset()
    cum_reward = 0
    n_episodes = 0
    n_timesteps = 0
    mean_reward = []
    goal_dists = []
    successes = []
    while n_episodes < args.test_episodes:
        action, _states = policy.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_reward += (policy.model.gamma**n_timesteps if args.discounted else 1)*reward

        if args.render:
            env.render()

        n_timesteps += 1

        if done:

            if 'goal_dist' in info:
                goal_dists.append(info['goal_dist'])
                print('Final distance from goal:', info['goal_dist'])

            if hasattr(config, 'performance_lb') and config.performance_lb is not None:
                successes.append(log_success(env, cum_reward, config))
            
            mean_reward.append(cum_reward)
            cum_reward = 0
            n_episodes += 1
            n_timesteps = 0
            obs = env.reset()

    print('Mean reward:', np.mean(mean_reward))
    print('Std reward:', np.std(mean_reward))
    if len(goal_dists) > 0:
        print('Mean distance to goal:', np.mean(goal_dists))
    if len(successes) > 0:
        print(f'Success rate: {round(np.array(successes).astype(int).mean() * 100, 2)}%')


def log_success(env, cum_reward, config):
    """Record whether the episode has been
        successfull or not"""
    if hasattr(env, 'success_metric'):
        metric = getattr(env, env.success_metric)
    else:
        metric = cum_reward

    return metric > config.performance_lb



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, type=str, help='Run path', required=True)
    parser.add_argument('--test_episodes', default=50, type=int, help='Test episodes')
    parser.add_argument('--best_on_target', default=False, action='store_true', help='If set, use best policy in terms of target success rate.')
    parser.add_argument('--discounted', default=False, action='store_true', help='Whether to compute the discounted return, with gamma inferred from the given run')
    parser.add_argument('--render', default=False, action='store_true', help='Render test episodes')
    parser.add_argument('--seed', default=None, type=int, help='Random seed')
    parser.add_argument('--device', default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--verbose', default=0, type=int, help='0,1,2')

    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
  main()

