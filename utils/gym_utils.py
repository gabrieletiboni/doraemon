"""Utility functions for handling and making gym environments"""
import pdb
import numpy as np
import gym
from gym.wrappers import FrameStack, FlattenObservation

from customwrappers.ObsToNumpy import ObsToNumpy
from customwrappers.ActionHistoryWrapper import ActionHistoryWrapper
from customwrappers.DynamicsInObs import DynamicsInObs
from customwrappers.AutoDRWrapper import AutoDRWrapper
from customwrappers.ReturnTrackerWrapper import ReturnTrackerWrapper
from customwrappers.ReturnDynamicsTrackerWrapper import ReturnDynamicsTrackerWrapper
from customwrappers.ReturnDynamicsTimestepsTrackerWrapper import ReturnDynamicsTimestepsTrackerWrapper


def is_locomotion_env(env_name):
    loco_envs = ['hopper', 'halfcheetah', 'walker', 'ant']

    # Exception
    if 'hopperhard' in env_name.lower():
        return False
        
    return np.any([loco_env in env_name.lower() for loco_env in loco_envs])

def make_rendering_env(env_name, env_kwargs):
    env = gym.make(env_name, **env_kwargs)
    env.set_verbosity(1)
    return env

def is_panda(env_name):
    return 'DMPandaPush-' in env_name

def get_panda_gym_args(args):
    args_names = ['qacc_factor', 'control_penalty_coeff', 'task_reward', 'search_space_id', 'absolute_acc_pen']
    kwargs = {}

    for name in args_names:
        if name in args:
            kwargs[name] = getattr(args, name)
    return kwargs

def get_env_kwargs(env_name, args):
    """Return env kwargs"""
    if is_panda(env_name):
        return get_panda_gym_args(args)
    else:
        return {}


def get_starting_beta_parameters(start_from_id, search_space_id, init_beta_param, args):
    """Get starting parameters a, b for beta given search_space_id.
        This function is for PandaPush envs only for now.
    """
    assert 'DMPandaPush-' in args.env, 'This function is for PandaPush envs only for now.'

    starting_env = gym.make(args.env, **{**args.env_kwargs, 'search_space_id': start_from_id})  # env with starting space
    starting_task = starting_env.get_task()

    wide_env = gym.make(args.env, **{**args.env_kwargs, 'search_space_id': search_space_id})  # wide target space
    wide_center_task = wide_env.get_task()
    lower_bounds = np.zeros(len(wide_center_task)) if is_locomotion_env(args.env) else None  # use zeros as lower_bounds for locomotion envs params
    dr_bounds = wide_env.get_uniform_dr_by_percentage(percentage=args.dr_percentage,
                                                      nominal_values=wide_center_task,
                                                      lower_bounds=lower_bounds,
                                                      dyn_mask=args.rand_only)

    bounds_stack = dr_bounds.reshape(len(wide_center_task), 2)
    bounds_widths = np.diff(bounds_stack).ravel()

    norm_starting_task = (starting_task-bounds_stack[:,0])*(1/bounds_widths)  # y = (x-m)/(M-m)
    
    norm_starting_variance = get_var_from_beta_params(init_beta_param, init_beta_param)

    # Get beta params a and b such that mean is `norm_starting_task` and var is `norm_starting_variance`
    mean, var = norm_starting_task, norm_starting_variance
    a = (  ((1-mean)/var) - (1/mean)   )*(mean**2)
    b = a*(1/mean - 1)

    return a, b

def get_var_from_beta_params(a,b):
    """Return variance of beta"""
    return (a*b) / ( ((a+b)**2) * (a+b+1) )


def make_wrapped_environment(env, args, wrapper=None):
    """Wrap env
        
        :param args.stack_history: int
                             number of previous obs and actions 
        :param args.rand_only: List[int]
                               dyn param indices mask
        :param args.dyn_in_obs: bool
                                condition the policy on the true dyn params
    """

    if args.stack_history is not None:
        env = FrameStack(env, args.stack_history+1)  # FrameStack considers the current obs as 1 stack
        env = ObsToNumpy(env)
        env = FlattenObservation(env)
        env = ActionHistoryWrapper(env, args.stack_history)

    if args.dyn_in_obs:
        env = DynamicsInObs(env, dynamics_mask=args.rand_only)

    if wrapper is not None:
        if wrapper == 'doraemon':
            env = ReturnDynamicsTrackerWrapper(env)
        elif wrapper == 'autodr':
            env = AutoDRWrapper(env, bound_sampling_prob=args.bound_sampling_prob)
        elif wrapper == 'lsdr':
            env = ReturnDynamicsTimestepsTrackerWrapper(env)
        elif wrapper == 'returnTracker':
            env = ReturnTrackerWrapper(env)

    return env

def get_actor_critic_obs_masks(args):
    """Return masks for actor and critic input observations.
    Used to give asymmetric information to actor/critic networks
    """
    assert not (args.actor_state_only and args.actor_history_only)
    actor_obs_mask, critic_obs_mask = None, None

    if args.actor_state_only:  # p(a|s) q(s,a,h)  OR  p(a|s) q(s,a,xi)
        """Mask obs as: obs[history_length*obs_dim : history_length*obs_dim + obs_dim]
        
            observation vector is layed out as:
                concatenate(state_history, s_t, action_history, (Optional)dynamics )
        """
        assert not args.algo == 'ppo', 'Asymmetric information is implemented for SAC only'
        assert args.stack_history is not None or args.dyn_in_obs is not False
        assert not (args.stack_history is not None and args.dyn_in_obs is not False)

        dummy_state_env = FlattenObservation(gym.make(args.env, **args.env_kwargs))
        flatten_state_dim = dummy_state_env.observation_space.shape[0]

        if args.stack_history is not None:  # p(a|s) q(s,a,h)
            actor_obs_mask = list(range(flatten_state_dim*args.stack_history, flatten_state_dim*args.stack_history + flatten_state_dim))
        else:  # p(a|s) q(s,a,xi)
            actor_obs_mask = list(range(0, flatten_state_dim))

        # Test masking to avoid errors later
        dummy_env = make_wrapped_environment(gym.make(args.env, **args.env_kwargs), args=args)
        dummy_ob = dummy_env.reset()
        dummy_ob_masked = dummy_ob[actor_obs_mask]


    elif args.actor_history_only:  # p(a|s,h), q(s,a,xi)
        """Mask obs as: obs[0 : history_length*obs_dim + obs_dim + action_history_dim]
        
            observation vector is layed out as:
                concatenate(state_history, s_t, action_history, (Optional)dynamics )
        """
        assert not args.algo == 'ppo', 'Asymmetric information is implemented for SAC only'
        assert args.stack_history is not None and args.dyn_in_obs is not False
        assert args.critic_dyn_only is True

        dummy_state_env = FlattenObservation(gym.make(args.env, **args.env_kwargs))
        flatten_state_dim = dummy_state_env.observation_space.shape[0]
        action_dim = dummy_state_env.action_space.shape[0]
        action_history_dim = args.stack_history*action_dim
        state_history_dim = args.stack_history*flatten_state_dim

        dummy_env = make_wrapped_environment(gym.make(args.env, **args.env_kwargs), args=args)
        dummy_ob = dummy_env.reset()
        dummy_ob_dim = dummy_ob.shape[0]
        dyn_dim = dummy_ob[state_history_dim + flatten_state_dim + action_history_dim:].shape[0]

        actor_obs_mask = list(range(0, state_history_dim + flatten_state_dim + action_history_dim))
        critic_obs_mask = list(range(state_history_dim, state_history_dim + flatten_state_dim)) + list(range(dummy_ob_dim-dyn_dim, dummy_ob_dim))

        # Test masking to avoid errors later
        dummy_ob_actor_masked = dummy_ob[actor_obs_mask]
        dummy_ob_critic_masked = dummy_ob[critic_obs_mask]

    return actor_obs_mask, critic_obs_mask


def get_dyn_pair_indexes_per_env(env_name):
    """Returns a tuple of two elements indicating which
    dynamics parameters to consider for visualization
    of 2D heatmaps
    """

    # Choose a single representative dynamics param pair
    # for each env for visualization
    dyn_pair_indexes_per_env = {
        'hopper': (0, 7),
        'halfcheetah': (0, 7),
        'walker2d': (0, 8),
        'cartpolehard': (0, 3),
        'cartpoleeasy': (0, 1),
        'swimmer': (3, 6),
        'ant': (0, 9),
        'reacher': (0, 3)
    }

    for key, pair in dyn_pair_indexes_per_env.items():
        if key in env_name.lower():
            return pair

    raise ValueError(f'Env name not found in dyn pair indexes: {env_name}')