"""
    Standalone custom implementation of Automatic Domain Randomization (AutoDR)
    (original paper at https://arxiv.org/abs/1910.07113)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import sys
import pdb
from copy import deepcopy
import time

import wandb
import numpy as np
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.beta import Beta
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, Bounds

from utils.utils import *
from utils.gym_utils import *
from policy.policy import Policy

class TrainingSubRtn():
    """Training subroutine wrapper
        
        Used by AutoDR class to handle
        training a policy with DR.
    """

    def __init__(self,
                 env,
                 algo,
                 lr,
                 gamma,
                 device,
                 seed,
                 actor_obs_mask,
                 critic_obs_mask,
                 n_eval_episodes,
                 run_path,
                 gradient_steps,
                 verbose=0,
                 eval_env=None):
        self.algo = algo
        self.lr = lr
        self.env = env
        self.eval_env = eval_env
        self.gamma = gamma
        self.seed = seed
        self.device = device
        self.run_path = run_path
        self.actor_obs_mask = actor_obs_mask
        self.critic_obs_mask = critic_obs_mask
        self.n_eval_episodes = n_eval_episodes
        self.gradient_steps = gradient_steps
        self.verbose = verbose

        self.agent = None


    def train(self,
              domainRandDistribution,
              max_training_steps,
              before_final_eval_cb=None,
              after_final_eval_cb=None,
              reset_agent=False):
        """Trains a policy until reward
            threshold is reached, or for a maximum
            number of steps.
        """
        self.env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get_params())
        self.env.set_dr_training(True)

        self.env.reset()

        # Turn on DR on eval_env as well
        self.eval_env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get_params())
        self.eval_env.set_dr_training(True)

        if self.agent is None or reset_agent:
            self.agent = Policy(algo=self.algo,
                                env=self.env,
                                lr=self.lr,
                                gamma=self.gamma,
                                device=self.device,
                                seed=self.seed,
                                actor_obs_mask=self.actor_obs_mask,
                                critic_obs_mask=self.critic_obs_mask,
                                gradient_steps=self.gradient_steps)

        mean_reward, std_reward, best_policy, which_one = self.agent.train(timesteps=max_training_steps,
                                                                           stopAtRewardThreshold=False,
                                                                           n_eval_episodes=self.n_eval_episodes,
                                                                           best_model_save_path=self.run_path,
                                                                           eval_env=self.eval_env,
                                                                           return_best_model=False,
                                                                           disable_eval=True,
                                                                           before_final_eval_cb=before_final_eval_cb,
                                                                           after_final_eval_cb=after_final_eval_cb)
        eff_timesteps = self.agent.model.num_timesteps
        self.env.set_dr_training(False)
        self.eval_env.set_dr_training(False)

        torch.save(best_policy, os.path.join(self.run_path, 'overall_best.pth'))

        return mean_reward, std_reward, best_policy, self.agent, eff_timesteps

    def reset(self):
        self.env.reset()
        return

    def eval(self, policy, domainRandDistribution, eval_episodes=None):
        """Evaluate policy with given DR distribution
            on self.env"""
        self.eval_env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get_params())
        self.eval_env.set_dr_training(True)
        agent = Policy(algo=self.algo,
                       env=self.eval_env,
                       device=self.device,
                       seed=self.seed,
                       actor_obs_mask=self.actor_obs_mask,
                       critic_obs_mask=self.critic_obs_mask)
        agent.load_state_dict(policy)

        mean_reward, std_reward = agent.eval(n_eval_episodes=(self.n_eval_episodes if eval_episodes is None else eval_episodes))
        self.eval_env.set_dr_training(False)

        return mean_reward, std_reward


    def reset_buffer(self, dim, low_or_high):
        self.env.env_method('reset_buffer', **{'dim': dim, 'low_or_high': low_or_high})

    def get_buffers(self, method_name='get_buffers'):
        """
            Collect all performance buffers from vectorized
            environments and aggregates them

            method_name: optionally specify a different method name to retrieve
                         the buffer, e.g. the custom success metric instead of returns.
        """
        individual_buffers = [item for item in self.env.env_method(method_name)]
        aggr_buffers = [[[], []] for i in range(len(individual_buffers[0]))]

        for curr_worker_buffer in individual_buffers:  # for each worker
            for i in range(len(curr_worker_buffer)):  # for each distr dimension
                aggr_buffers[i][0] += curr_worker_buffer[i][0]
                aggr_buffers[i][1] += curr_worker_buffer[i][1]

        return aggr_buffers



class DomainRandDistribution():
    """Handles Domain Randomization distributions"""

    def __init__(self,
                 distr: List[Dict]):
        self.set(distr)
        return

    # Methods to override in child envs:
    # ----------------------------
    def set(self, distr):   
        raise NotImplementedError
    def sample(self, n_samples=1):
        raise NotImplementedError
    def sample_univariate(self, i, n_samples=1):
        raise NotImplementedError
    def _univariate_pdf(self, x, i, log=False, to_distr=None, standardize=False):
        raise NotImplementedError
    def pdf(self, x, log=False, requires_grad=False, standardize=False, to_params=None):
        raise NotImplementedError
    def _standardize_value(self, x):
        raise NotImplementedError
    def kl_divergence(self, q, requires_grad=False, p_params=None, q_params=None):
        raise NotImplementedError
    def entropy(self, standardize=False):
        raise NotImplementedError
    def _to_distr_with_grad(self, p, to_params=None):
        raise NotImplementedError
    def update_parameters(self, params):
        raise NotImplementedError
    def support(self, i):
        """Return support in dimension i"""
        raise NotImplementedError
    def visualize_distr(self, ax=None, **plot_kwargs):
        raise NotImplementedError
    def print(self):
        raise NotImplementedError
    def to_string(self):
        raise NotImplementedError
    # ----------------------------

    def get_dr_type(self):
        return self.dr_type

    def get(self):
        return self.distr

    def get_stacked_params(self):
        return self.parameters.detach().numpy()

    def get_params(self):
        return self.parameters


class BetaDistribution(DomainRandDistribution):
    """Beta Domain Randomization distribution"""

    def __init__(self,
                 distr: List[Dict]):
        DomainRandDistribution.__init__(self,
                                        distr)

        self.dr_type = 'beta'
        return

    def set(self, distr):
        """
            distr: list of dict
                   4 keys per dimensions are expected:
                    m=min, M=max, a, b

                Y ~ Beta(a,b,m,M)
                y = x(M-m) + m
                f(y) = f_x((y-m)/(M-m))/(M-m)
        """
        self.distr = distr.copy()
        self.ndims = len(self.distr)
        self.to_distr = []
        self.parameters = torch.zeros((self.ndims*2), dtype=torch.float32)
        for i in range(self.ndims):
            self.parameters[i*2] = float(distr[i]['a'])
            self.parameters[i*2 + 1] = float(distr[i]['b'])
            self.to_distr.append(Beta(self.parameters[i*2], self.parameters[i*2 + 1]))

    def entropy(self, standardize=False):
        """Returns entropy of distribution"""
        entropy = 0
        for i in range(self.ndims):
            if standardize:
                entropy += self.to_distr[i].entropy()
            else:
                # Y = aX + b => H(Y) = H(X) + log(a) 
                m, M = self.distr[i]['m'], self.distr[i]['M']
                entropy += self.to_distr[i].entropy() + torch.log(torch.tensor(M-m))

        return entropy

class UniformDistribution(DomainRandDistribution):
    """Uniform Domain Randomization distribution"""

    def __init__(self,
                 distr: List[Dict]):
        DomainRandDistribution.__init__(self,
                                        distr)

        self.dr_type = 'uniform'
        return

    def set(self, distr):   
        """Sets distribution"""
        self.distr = distr.copy()
        self.ndims = len(self.distr)
        self.to_distr = []
        self.parameters = torch.zeros((self.ndims*2), dtype=torch.float64)
        for i in range(self.ndims):
            self.parameters[i*2] = float(distr[i]['m'])
            self.parameters[i*2 + 1] = float(distr[i]['M'])
            self.to_distr.append(Uniform(self.parameters[i*2], self.parameters[i*2 + 1]))

    def update_parameters(self, params):
        """Update the current uniform parameters"""
        distr = deepcopy(self.distr)
        for i in range(self.ndims):
            distr[i]['m'] = params[i*2]
            distr[i]['M'] = params[i*2 + 1]

        self.set(distr=distr)

    def entropy(self, standardize=False):
        """Returns entropy of distribution"""
        entropy = 0
        for i in range(self.ndims):
            # if standardize:
            entropy += self.to_distr[i].entropy()
            # else:
            #     # Y = aX + b => H(Y) = H(X) + log(a) 
            #     m, M = self.distr[i]['m'], self.distr[i]['M']
            #     entropy += self.to_distr[i].entropy() + torch.log(torch.tensor(M-m))

        return entropy

    def print(self):
        for i in range(self.ndims):
            print(f"dim{i}: [{round(self.distr[i]['m'], 3)} {round(self.distr[i]['M'], 3)}]")

    def to_string(self):
        string = ''
        for i in range(self.ndims):
            string += f"dim{i}: [{round(self.distr[i]['m'], 3)} {round(self.distr[i]['M'], 3)}] | "
        return string

    def set_standardized_uniform(self,
                                 uniform_bounds : DomainRandDistribution):
        """Updates distribution to a Uniform Standardized
        distribution w.r.t uniform_bounds distribution
        
            uniform_bounds: relative bounds for standardization.
                           current distribution MUST be within
                           the uniform_bounds
        """
        assert self._is_support_within(uniform_bounds)

        distr = []

        for i in range(self.ndims):
            x_m = self.distr[i]['m']
            std_x_m = (x_m - uniform_bounds.distr[i]['m']) / (uniform_bounds.distr[i]['M'] - uniform_bounds.distr[i]['m'])

            x_M = self.distr[i]['M']
            std_x_M = (x_M - uniform_bounds.distr[i]['m']) / (uniform_bounds.distr[i]['M'] - uniform_bounds.distr[i]['m'])

            distr.append({'m': std_x_m, 'M': std_x_M})

        self.set(distr=distr)

    def get_standardized_uniform(self,
                                 uniform_bounds : DomainRandDistribution):
        """Returns a Uniform distribution standardized in [0, 1]
            w.r.t given uniform_bounds
        
            uniform_bounds: relative bounds for standardization.
                           current distribution MUST be within
                           the uniform_bounds
        """
        assert self._is_support_within(uniform_bounds)

        distr = []

        for i in range(self.ndims):
            x_m = self.distr[i]['m']
            std_x_m = (x_m - uniform_bounds.distr[i]['m']) / (uniform_bounds.distr[i]['M'] - uniform_bounds.distr[i]['m'])

            x_M = self.distr[i]['M']
            std_x_M = (x_M - uniform_bounds.distr[i]['m']) / (uniform_bounds.distr[i]['M'] - uniform_bounds.distr[i]['m'])

            distr.append({'m': std_x_m, 'M': std_x_M})

        return UniformDistribution(distr)



    def _is_support_within(self,
                           uniform_bounds : DomainRandDistribution):
        """Checks whether current support is within some bounds"""
        for i in range(self.ndims):
            if self.distr[i]['m'] < uniform_bounds.distr[i]['m']:
                return False
            elif self.distr[i]['M'] > uniform_bounds.distr[i]['M']:
                return False
        return True


class AutoDR():
    """Automatic Domain Randomization
       https://arxiv.org/abs/1910.07113
       
       Custom implementation, following the
       original paper. This implementation uses stable-baselines3
       vectorized environments for the parallelization of
       the ADR Eval Worker and the Rollout Worker.
    """

    def __init__(self,
                 training_subrtn: TrainingSubRtn,
                 init_distr: UniformDistribution,
                 boundaries: UniformDistribution,
                 delta: float,
                 performance_threshold_high: float,
                 performance_threshold_low: float,
                 check_update_freq: int,
                 buffer_size: int,
                 budget: float = 5000000,
                 test_episodes=100,
                 train_until=True,
                 force_success_with_returns=False,
                 training_subrtn_kwargs={},
                 verbose=0):
        """
            training_subrtn: handles the RL training subroutine
            budget: total number of env steps allowed.
            delta: update step size (normalized w.r.t. [0, 1] max boundaries).
                   i.e., delta is the percentage step size w.r.t the boundaries.
            performance_threshold_high/low: thresholds for updating
                                            the distribution
            init_distr: starting distribution
            boundaries: maximum bounds allowed for each uniform dimension
            buffer_size: length of performance data buffers
            check_update_freq: check whether any buffer is full every < > timesteps
            force_success_with_returns: force using returns for computing average performance,
                                        instead of custom metric defined in the env
                                        (e.g. env.success_metric)
        """
        self.training_subrtn = training_subrtn
        self.training_subrtn_kwargs = training_subrtn_kwargs
        self.check_update_freq = check_update_freq
        self.test_episodes = test_episodes
        self.budget = budget
        self.init_budget = budget
        self.buffer_size = buffer_size

        self.delta = delta
        self.performance_threshold_low = performance_threshold_low
        self.performance_threshold_high = performance_threshold_high 

        self.init_distr = deepcopy(init_distr)
        self.current_distr = init_distr
        self.boundaries = boundaries
        self.train_until = train_until
        self.force_success_with_returns = force_success_with_returns
        self.verbose = verbose
        
        self.min_training_steps = 100
        self.current_iter = 0
        self.distr_history = []
        self.final_policy = None
        # self.best_policy = None
        self.best_policy_return = -np.inf
        self.best_policy_succ_rate = -1
        self.best_distr = None
        self.best_iter = None


    def dummy_autodr_update(self):
        curr_entropy = self.current_distr.entropy().item()
        wandb.log({"entropy": curr_entropy, 'timestep': self.current_timestep})

    def learn(self,
              ckpt_dir: str = None):
        """
            Run AutoDR

            ckpt_dir : path for saving checkpoints
        """
        self.distr_history.append(deepcopy(self.current_distr))
        distr_has_changed = False
        print('Init entropy:', self.current_distr.entropy().item())
        wandb.log({"entropy": self.current_distr.entropy().item(), 'timestep': 0})
        print('Maximum entropy achievable:', self.boundaries.entropy().item())
        trained_until_flag = False
        if self.train_until and self.verbose >= 1:
            print(f'Training with fixed DR distribution until performance threshold high={self.performance_threshold_high} is reached.')


        while self.is_there_budget_for_iter():
            """
                1. Call the training sub routine for a fixed number of timesteps
            """
            start = time.time()
            training_budget = min(self.budget, self.check_update_freq)
            reward, std_reward, policy, agent, eff_timesteps = self.training_subrtn.train(domainRandDistribution=self.current_distr,
                                                                                          max_training_steps=training_budget,
                                                                                          before_final_eval_cb=self.before_final_eval_cb,
                                                                                          after_final_eval_cb=self.after_final_eval_cb,
                                                                                          **self.training_subrtn_kwargs)
            self.budget -= eff_timesteps  # decrease remaining budget
            individual_returns = self._flatten(self.training_subrtn.eval_env.env_method('get_buffer'))  # retrieve returns during final eval
            individual_succmetrics = self._flatten(self.training_subrtn.eval_env.env_method('get_succ_metric_buffer'))  # retrieve custom metric during final eval
            
            # Use custom metric for defining success rate, if defined
            if len(individual_succmetrics) > 0 and not self.force_success_with_returns:
                individual_metrics = individual_succmetrics.copy()
            else:
                individual_metrics = individual_returns.copy()

            train_success_rate = torch.tensor(individual_metrics >= self.performance_threshold_high, dtype=torch.float32).mean()

            wandb.log({"budget": self.budget, 'iter': self.current_iter})
            wandb.log({"train_mean_reward": reward, 'timestep': self.current_timestep})
            wandb.log({"train_stdev_reward": std_reward, 'timestep': self.current_timestep})
            wandb.log({"train_success_rate": train_success_rate, "timestep": self.current_timestep})
            if len(individual_succmetrics) > 0:
                wandb.log({"train_mean_succ_metric": individual_succmetrics.mean(), "timestep": self.current_timestep})
            if self.verbose >= 2:
                print('---')
                print(f"TIME - policy training (s): {round(time.time() - start, 2)}")
                print(f'Train mean reward: {reward} +- {std_reward} (ts {self.current_timestep} / {self.init_budget})')
                print(f'Train succ rate: {train_success_rate} (ts {self.current_timestep} / {self.init_budget})')
                if len(individual_succmetrics) > 0:
                    print(f'Train mean succ metric: {individual_succmetrics.mean()} (ts {self.current_timestep} / {self.init_budget})')
                print(f'Budget stats: {eff_timesteps} (used in last iteration) | {self.budget} (remaining)')
            
            # Test current policy on max entropy distribution
            start = time.time()
            self.test_on_target_distr(policy, ckpt_dir)
            if self.verbose >= 2:
                print(f"policy eval time on target distr (s): {round(time.time() - start, 2)}")

            # Save checkpoint
            if ckpt_dir is not None:
                additional_keys = {'mean_reward': reward, 'std_reward': std_reward, 'best_policy_succ_rate': self.best_policy_succ_rate, 'best_policy_return': self.best_policy_return, 'policyfilename': 'overall_best.pth'}
                self._save_checkpoint(save_dir=ckpt_dir, additional_keys=additional_keys)

            if self.train_until and trained_until_flag is False:
                if reward >= self.performance_threshold_high:
                    if self.verbose >= 1:
                        print(f'--- Performance threshold achieved for the first time. AutoDR can now start updating the DR distribution.')
                    trained_until_flag = True
                else:
                    self._reset_buffers()
                    self.dummy_autodr_update()
                    continue

            """
                2. Update distribution (if any buffer has been filled)
            """
            start = time.time()
            buffers = self._get_buffers()

            succ_metrics = self._get_succ_metric_buffers()  # retrieve custom metric for computing average performance collected during training (optional)

            # Replace returns with the custom metric, if defined
            if not self._is_buffer_empty(succ_metrics) and not self.force_success_with_returns:
                buffers = succ_metrics

            dim_changed = []
            for dim, buffer_dim in enumerate(buffers):
                if self._update_low_buffer(buffer_dim[0], dim=dim) or \
                   self._update_high_buffer(buffer_dim[1], dim=dim):
                    distr_has_changed = True
                    dim_changed.append(dim)

            if self.verbose >= 2:
                print(f"TIME - buffers check and update (s): {round(time.time() - start, 2)}")
                print('---')

            if distr_has_changed:
                self.distr_history.append(deepcopy(self.current_distr))
                self.current_iter += 1
                distr_has_changed = False

                print(f'\n\n=== AutoDR Step {self.current_iter}')
                print(f'current distribution params: {self.current_distr.get_stacked_params()}')

                curr_entropy = self.current_distr.entropy().item()
                wandb.log({"entropy": curr_entropy, 'timestep': self.current_timestep})
                if self.verbose >= 1:
                    print('Entropy after update:', curr_entropy)
                    print('Dimensions changed:', dim_changed)
            else:
                curr_entropy = self.current_distr.entropy().item()
                wandb.log({"entropy": curr_entropy, 'timestep': self.current_timestep})


        
        self.final_policy = deepcopy(policy)
        return

    def _update_low_buffer(self, buffer:List, dim:int):
        if self._is_buffer_full(buffer):
            avg_performance = np.mean(buffer)
            self._reset_buffer(dim, 0)

            global_min = self.boundaries.get_params()[dim*2]
            step_size = self._get_denormalized_delta(self.delta, self.boundaries, dim=dim) 

            if avg_performance >= self.performance_threshold_high:
                ### Widen distribution: decrease low bound by delta
                params = np.copy(self.current_distr.get_params())
                
                # Check if lower bound is already at min possible value
                if params[dim*2] == global_min:
                    return False
                
                if params[dim*2] - step_size < global_min:
                    new_val = global_min
                else:
                    new_val = params[dim*2] - step_size

                params[dim*2] = new_val
                self.current_distr.update_parameters(params)
                # print(f'--- Dim {dim} bound low widened, perf is above the threshold.')
                return True                
            elif avg_performance <= self.performance_threshold_low:
                ### Shorten distribution: increase low bound by delta
                params = np.copy(self.current_distr.get_params())
                
                # Check if lower bound is already equal to the upper bound
                if params[dim*2] >= params[dim*2 + 1] - 1e-5:
                    return False
                
                if params[dim*2] + step_size < params[dim*2 + 1] - 1e-5:
                    new_val = params[dim*2] + step_size
                else:
                    new_val = params[dim*2 + 1] - 1e-5

                params[dim*2] = new_val
                self.current_distr.update_parameters(params)
                # print(f'--- Dim {dim} bound low shortened, perf is too low.')
                return True

        return False

    def _update_high_buffer(self, buffer:List, dim:int):
        if self._is_buffer_full(buffer):
            avg_performance = np.mean(buffer)
            self._reset_buffer(dim, 1) 

            global_max = self.boundaries.get_params()[dim*2 + 1]
            step_size = self._get_denormalized_delta(self.delta, self.boundaries, dim=dim) 

            if avg_performance >= self.performance_threshold_high:
                ### Widen distribution: increase high bound by delta
                params = np.copy(self.current_distr.get_params())
                
                # Check if high bound is already at max possible value
                if params[dim*2 + 1] == global_max:
                    return False
                
                if params[dim*2 + 1] + step_size > global_max:
                    new_val = global_max
                else:
                    new_val = params[dim*2 + 1] + step_size

                params[dim*2 + 1] = new_val
                self.current_distr.update_parameters(params)
                # print(f'--- Dim {dim} bound high widened, perf is above the threshold.')
                return True        
                
                return True                
            elif avg_performance <= self.performance_threshold_low:
                ### Shorten distribution: decrease high bound by delta
                params = np.copy(self.current_distr.get_params())
                
                # Check if lower bound is already equal to the upper bound
                if params[dim*2 + 1] <= params[dim*2] + 1e-5:
                    return False
                
                if params[dim*2 + 1] - step_size > params[dim*2] + 1e-5:
                    new_val = params[dim*2 + 1] - step_size
                else:
                    new_val = params[dim*2] + 1e-5

                params[dim*2 + 1] = new_val
                self.current_distr.update_parameters(params)
                # print(f'--- Dim {dim} bound high shortened, perf is too low.')
                return True

                return True
        return False

    def _get_denormalized_delta(self, delta: float, boundaries: UniformDistribution, dim: int):
        """Denormalize delta step from [0,1] to
        actual step w.r.t. current dimension"""
        m, M = boundaries.get_params()[[dim*2, dim*2 + 1]]
        return (M-m)*delta

    def test_on_target_distr(self, policy, ckpt_dir=None):
        """Evaluate policy on target distribution and
            keep track of best overall policy found

            ckpt_dir: save best in this dir if provided
        """
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': True})  # track episode returns
        self.training_subrtn.eval_env.env_method('reset_buffer')  # reset previous returns
        target_mean_reward, target_std_reward = self.training_subrtn.eval(policy, self.boundaries, eval_episodes=self.test_episodes)
        
        # returns = self.training_subrtn.eval_env.env_method('get_buffer')  # retrieve tracked returns
        # returns = self._flatten(returns)
        returns = np.array(self._flatten(self.training_subrtn.eval_env.env_method('get_buffer')))  # retrieve tracked returns
        succ_metrics = np.array(self._flatten(self.training_subrtn.eval_env.env_method('get_succ_metric_buffer')))  # retrieve custom metric for measuring success, and flatten values

        if len(succ_metrics) == 0 or self.force_success_with_returns:
            # Use returns for measuring success
            target_success_rate = torch.tensor(returns >= self.performance_threshold_high, dtype=torch.float32).mean()
        else:
            # Use custom metric for measuring success
            target_success_rate = torch.tensor(succ_metrics >= self.performance_threshold_high, dtype=torch.float32).mean()

        # target_success_rate = torch.tensor(returns >= self.performance_threshold_high, dtype=torch.float32).mean()
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': False})
        self.training_subrtn.eval_env.env_method('reset_buffer')  # just in case

        if target_success_rate > self.best_policy_succ_rate:
            self.best_policy_return = target_mean_reward
            self.best_policy_succ_rate = target_success_rate
            # self.best_policy = deepcopy(policy)
            self.best_distr = deepcopy(self.current_distr)
            self.best_iter = self.current_iter
            wandb.run.summary[f'best_distr'] = self.current_distr.to_string()
            wandb.run.summary[f'best_iter'] = self.current_iter

            if ckpt_dir is not None:
                torch.save(policy, os.path.join(ckpt_dir, 'best_on_target.pth'))
        if self.verbose >= 2:
            print(f'Target mean reward at iter {self.current_iter}: {target_mean_reward} +- {target_std_reward}')
            print(f'Target success rate at iter {self.current_iter}: {target_success_rate}')
        wandb.log({"target_mean_reward": target_mean_reward, 'timestep': self.current_timestep})
        wandb.log({"target_success_rate": target_success_rate, 'timestep': self.current_timestep})
        if len(succ_metrics) > 0:
            wandb.log({"target_mean_succ_metric": np.mean(succ_metrics), "timestep": self.current_timestep})


    def _flatten(self, multi_list):
        """Flatten a list of lists with potentially
        different lenghts into a 1D np array"""
        flat_list = [] 
        for single_list in multi_list:
            flat_list += single_list

        return np.array(flat_list, dtype=np.float64)

    def before_final_eval_cb(self):
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': True})  # track episode returns
        self.training_subrtn.eval_env.env_method('reset_buffer')  # reset previous returns

    def after_final_eval_cb(self):
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': False})

    def _is_buffer_full(self, buffer: List):
        return len(buffer) >= self.buffer_size

    def _is_buffer_empty(self, buffer: List):
        lengths = np.sum([len(dim_buffer[0]) + len(dim_buffer[1]) for dim_buffer in buffer])
        return lengths == 0

    def is_there_budget_for_iter(self):
        return self.budget > self.min_training_steps

    def _reset_buffers(self):
        buffers = self._get_buffers()
        for dim, _ in enumerate(buffers):
            self.training_subrtn.reset_buffer(dim, 0)
            self.training_subrtn.reset_buffer(dim, 1)

    def _reset_buffer(self, dim, low_or_high):
        self.training_subrtn.reset_buffer(dim, low_or_high)

    def _get_buffers(self):
        return self.training_subrtn.get_buffers(method_name='get_buffers')

    def _get_succ_metric_buffers(self):
        return self.training_subrtn.get_buffers(method_name='get_succ_metric_buffers')

    def _save_checkpoint(self, save_dir, additional_keys: Dict = {}):
        """Save AutoDR checkpoint"""
        checkpoint = {
                        'iter': self.current_iter,
                        'distr': self.current_distr,
                        'distr_history': self.distr_history,
                        'budget': self.budget,
                        'current_timestep': self.current_timestep
                     }

        # Append additional data 
        checkpoint = {**checkpoint, **additional_keys}

        save_object(checkpoint, save_dir=save_dir, filename='last_checkpoint')

    @property
    def current_timestep(self):
        """
            return the current number of training timesteps consumed
        """
        return self.init_budget - self.budget