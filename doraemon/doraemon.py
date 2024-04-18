"""Domain Randomization via Entropy Maximization (DORAEMON)"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import sys
import pdb
from copy import deepcopy
import time

import wandb
import numpy as np
import torch
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, Bounds
import scipy.stats as st

from utils.utils import *
from utils.gym_utils import *
from policy.policy import Policy

class TrainingSubRtn():
    """Training subroutine wrapper
        
        Used by DORAEMON class to handle
        training in between iterations, retrieving the
        trained policy and sampler, and others.
    """

    def __init__(self,
                 env,
                 eval_env,
                 algo,
                 lr,
                 gamma,
                 device,
                 seed,
                 actor_obs_mask,
                 critic_obs_mask,
                 n_eval_episodes,
                 eval_freq,
                 run_path,
                 gradient_steps,
                 verbose=0):
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
        self.eval_freq = eval_freq
        self.gradient_steps = gradient_steps
        self.verbose = verbose

        self.agent = None


    def train(self,
              domainRandDistribution,
              doraemon_iter,
              performance_threshold,
              max_training_steps,
              stopAtRewardThreshold,
              reset_agent=False):
        """Trains a policy until reward
            threshold is reached, or for a maximum
            number of steps.
        """
        self.env.env_method('set_expose_episode_stats', **{'flag': True})
        self.reset_buffer()
        self.env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get())
        self.env.set_dr_training(True)

        # Turn on DR on eval_env as well
        self.eval_env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get())
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
                                                                           stopAtRewardThreshold=stopAtRewardThreshold,
                                                                           reward_threshold=performance_threshold,
                                                                           n_eval_episodes=self.n_eval_episodes,
                                                                           eval_freq=self.eval_freq,
                                                                           best_model_save_path=self.run_path,
                                                                           eval_env=self.eval_env,
                                                                           return_best_model=False,
                                                                           wandb_loss_suffix=f'_iter{doraemon_iter}')
        eff_timesteps = self.agent.model.num_timesteps

        self.eval_env.set_dr_training(False)
        self.env.set_dr_training(False)
        self.env.env_method('set_expose_episode_stats', **{'flag': False})

        torch.save(best_policy, os.path.join(self.run_path, 'overall_best.pth'))

        return mean_reward, std_reward, best_policy, self.agent, eff_timesteps

    def reset(self):
        self.env.reset()
        return

    def eval(self, policy, domainRandDistribution, eval_episodes=None):
        """Evaluate policy with given DR distribution
            on self.env"""
        self.eval_env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get())
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

    def get_buffer(self):
        """Retrieves episode statistics (dynamics, return)
           from each training env
        """
        dynamics = []
        returns = [] 

        for buffer in self.env.env_method('get_buffer'):
            for episode_stats in buffer:
                dynamics.append(episode_stats['dynamics'])
                returns.append(episode_stats['return'])

        return np.array(dynamics, dtype=np.float64), np.array(returns)

    def get_succ_metric_buffer(self):
        """Retrieves custom metric used for computing success.
            E.g. distance from target location in a pushing task.
        """
        return np.concatenate(self.env.env_method('get_succ_metric_buffer')).ravel()

    def reset_buffer(self):
        self.env.env_method('reset_buffer')


class DomainRandDistribution():
    """Handles Domain Randomization distributions"""

    def __init__(self,
                 dr_type: str,
                 distr: List[Dict]):
        self.set(dr_type, distr)
        return

    def set(self, dr_type, distr):
        if dr_type == 'beta':
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
        else:
            raise Exception('Unknown dr_type:'+str(dr_type))
        
        self.dr_type = dr_type
        return

    def sample(self, n_samples=1):
        if self.dr_type == 'beta':
            values = []
            for i in range(self.ndims):
                m, M = self.distr[i]['m'], self.distr[i]['M']
                values.append(self.to_distr[i].sample(sample_shape=(n_samples,)).numpy()*(M - m) + m)
            return np.array(values).T

    def sample_univariate(self, i, n_samples=1):
        if self.dr_type == 'beta':
            values = []
            m, M = self.distr[i]['m'], self.distr[i]['M']
            values.append(self.to_distr[i].sample(sample_shape=(n_samples,)).numpy())  # *(M - m) + m
            return np.array(values).T

    def _univariate_pdf(self, x, i, log=False, to_distr=None, standardize=False):
        """
            Computes univariate pdf(value) for
            i-th independent variable

            to_distr: custom torch univariate distribution list
            standardize: compute beta pdf in standard interval [0, 1]
        """
        to_distr = self.to_distr if to_distr is None else to_distr

        if self.dr_type == 'beta':
            m, M = self.distr[i]['m'], self.distr[i]['M']
            if np.isclose(M-m, 0):
                return np.isclose(x, m).astype(int)  # 1 if x = m = M, 0 otherwise
            else:
                if log:
                    if standardize:
                        return to_distr[i].log_prob(torch.tensor(x))
                    else:
                        return to_distr[i].log_prob(torch.tensor((x-m)/(M-m))) - torch.log(torch.tensor(M-m))
                else:
                    if standardize:
                        return torch.exp(to_distr[i].log_prob(torch.tensor(x)))
                    else:
                        return torch.exp(to_distr[i].log_prob(torch.tensor((x-m)/(M-m))))/(M-m)

        return

    def pdf(self, x, log=False, requires_grad=False, standardize=False, to_params=None):
        """
            Computes pdf(x)

            x: torch.tensor (Batch x ndims)
            log: compute the log(pdf(x))
            requires_grad: keep track of gradients w.r.t. beta params
            standardize: compute pdf in the standard [0, 1] interval,
                         by rescaling the input value
        """
        assert len(x.shape) == 2, 'Input tensor is expected with dims (batch, ndims)'
        density = torch.zeros(x.shape[0]) if log else torch.ones(x.shape[0])
        custom_to_distr = None
        if requires_grad:
            custom_to_distr, to_params = self._to_distr_with_grad(self, to_params=to_params)

        if standardize:
            x = self._standardize_value(x)

        for i in range(self.ndims):
            if log:
                density += self._univariate_pdf(x[:, i], i, log=True, to_distr=custom_to_distr, standardize=standardize)
            else:
                density *= self._univariate_pdf(x[:, i], i, log=False, to_distr=custom_to_distr, standardize=standardize)

        if requires_grad:
            return density, to_params
        else:
            return density

    def _standardize_value(self, x):
        """Linearly scale values from [m, M] to [0, 1]"""
        norm_x = x.copy()
        for i in range(self.ndims):
            m, M = self.distr[i]['m'], self.distr[i]['M']
            norm_x[:, i] =  (x[:, i] - m) / (M - m)
        return norm_x

    def kl_divergence(self, q, requires_grad=False, p_params=None, q_params=None):
        """Returns KL_div(self || q)

            q: DomainRandDistribution
            requires_grad: compute computational graph w.r.t.
                           beta parameters
        """
        assert isinstance(q, DomainRandDistribution)
        assert self.dr_type == q.dr_type 
        
        if self.dr_type == 'beta':
            if requires_grad:
                p_distr, p_params = self._to_distr_with_grad(self, to_params=p_params)
                q_distr, q_params = self._to_distr_with_grad(q,    to_params=q_params)
            else:
                p_distr = self.to_distr
                q_distr = q.to_distr

            kl_div = 0
            for i in range(self.ndims):
                # KL does not depend on loc params [m, M]
                kl_div += torch.distributions.kl_divergence(p_distr[i], q_distr[i])

            if requires_grad:
                return kl_div, p_params, q_params
            else:
                return kl_div

    def entropy(self, standardize=False):
        """Returns entropy of distribution"""
        if self.dr_type == 'beta':
            entropy = 0
            for i in range(self.ndims):
                if standardize:
                    entropy += self.to_distr[i].entropy()
                else:
                    # Y = aX + b => H(Y) = H(X) + log(a) 
                    m, M = self.distr[i]['m'], self.distr[i]['M']
                    entropy += self.to_distr[i].entropy() + torch.log(torch.tensor(M-m))

            return entropy

    def _to_distr_with_grad(self, p, to_params=None):
        """
            Returns list of torch Beta distributions
            given a DomainRandDistribution object p
        """
        if to_params is None:
            params = p.get_stacked_params()
            to_params = torch.tensor(params, requires_grad=True)

        to_distr = []
        for i in range(self.ndims):
            to_distr.append(Beta(to_params[i*2], to_params[i*2 + 1]))
        return to_distr, to_params

    def update_parameters(self, params):
        """Update the current beta parameters"""
        if self.dr_type == 'beta':
            distr = deepcopy(self.distr)
            for i in range(self.ndims):
                distr[i]['a'] = params[i*2]
                distr[i]['b'] = params[i*2 + 1]

            self.set(dr_type=self.get_dr_type(), distr=distr)

    def get(self):
        return self.distr

    def get_stacked_bounds(self):
        return np.array([[item['m'], item['M']] for item in self.distr]).reshape(-1)

    def get_stacked_params(self):
        return self.parameters.detach().numpy()

    def get_params(self):
        return self.parameters

    def get_dr_type(self):
        return self.dr_type

    def visualize_distr(self, ax=None, only_dims=None, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=self.ndims, figsize=(8,5))
            assert only_dims is None

        axes = [ax] if not isinstance(ax, np.ndarray) else ax  # handle case of ax not being list if it's single figure/dim
        only_dims = only_dims if only_dims is not None else list(range(self.ndims))  # include all dimensions
        for j, i in enumerate(only_dims):
            x = np.linspace(self.distr[i]['m'], self.distr[i]['M'], 100)
            axes[j].plot(x, self._univariate_pdf(x, i), **{'lw': 3, 'alpha':0.6, 'label': f'beta pdf dim{i}', **plot_kwargs})

    def print(self):
        if self.dr_type == 'beta':
            for i in range(self.ndims):
                print(f'dim{i}:', self.distr[i])

    def to_string(self):
        string = ''
        if self.dr_type == 'beta':
            for i in range(self.ndims):
                string += f"dim{i}: {self.distr[i]} | "
        return string

    @staticmethod
    def beta_from_stacked(stacked_bounds: np.ndarray, stacked_params: np.ndarray):
        """Creates instance of this class from the given stacked
            array of parameters

            stacked_bounds: beta boundaries [m_1, M_1, m_2, M_2, ...]
            stacked_params: beta parameters [a_1, b_1, a_2, b_2, ...]
        """
        distr = []
        ndim = stacked_bounds.shape[0]//2
        for i in range(ndim):
            d = {}
            d['m'] = stacked_bounds[i*2]
            d['M'] = stacked_bounds[i*2 + 1]
            d['a'] = stacked_params[i*2]
            d['b'] = stacked_params[i*2 + 1]
            distr.append(d)
        return DomainRandDistribution(dr_type='beta', distr=distr)

    @staticmethod
    def sigmoid(x, lb=0, up=1):
        """sigmoid of x"""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        sig = (up-lb)/(1+torch.exp(-x)) + lb
        return sig

    @staticmethod
    def inv_sigmoid(x, lb=0, up=1):
        """return sigmoid^-1(x)"""
        x = x if torch.is_tensor(x) else torch.tensor(x)
        assert torch.all(x <= up) and torch.all(x >= lb)
        inv_sig = -torch.log((up-lb)/(x-lb) - 1)
        return inv_sig


class DORAEMON():
    """Domain Randomization via Entropy Maximization"""
    def __init__(self,
                 training_subrtn: TrainingSubRtn,
                 performance_lower_bound: float,
                 kl_upper_bound: float,
                 init_distr: DomainRandDistribution,
                 target_distr: DomainRandDistribution,
                 budget: float = 5000000,
                 max_training_steps: int = 100000,
                 stopAtRewardThreshold: bool = False,
                 reward_threshold_margin: float = 0.,
                 bootstrap_values: bool = False,
                 min_dynamics_samples=100,
                 max_dynamics_samples=1000,
                 reset_agent=False,
                 test_episodes=100,
                 train_until_performance_lb=False,
                 hard_performance_constraint=False,
                 robust_estimate=False,
                 alpha_ci=0.9,
                 performance_lb_percentile=None,
                 success_rate_condition=None,
                 prior_constraint=False,
                 force_success_with_returns=False,
                 init_beta_param=100.,
                 beta_param_bounds=None,
                 training_subrtn_kwargs={},
                 verbose=0):
        """
            training_subrtn: handles the RL training subroutine
            performance_lower_bound: J constraint in DORAEMON's opt. problem
            kl_upper_bound: KL constraint in DORAEMON'S opt. prblem
            init_distr: starting distribution
            target_distr: target distribution to converge to
            stopAtRewardThreshold: stop the RL training suboutine when 
                                   performance_lower_bound + margin is reached
            reward_threshold_margin: margin for early RL training stopping.
            budget: total number of env steps allowed.
            max_training_steps: training steps for the RL training subroutine
            bootstrap_values: compute J constraint with MC samples [False],
                              or bootstrapped samples given an estimate
                              of the context-aware value function V(s, xi) [True] 
            min_dynamics_samples: (DEPRECATED) min num of dynamics samples for computing J constraint
            max_dynamics_samples: (DEPRECATED) max num of dynamics samples for computing J constraint
            reset_agent: whether to reset the sb3 training agent (i.e. reset replay buffer, ...)
            train_until_performance_lb: bool, do not change the distribution
                                        until the constraint is satisfied the first time.
            hard_performance_constraint: bool, performance constraint in opt. problem is hard
            robust_estimate: bool, use the estimated lower-confidence-bound as
                             performance constraint instead of sample mean.
            alpha_ci: float, confidence level of robust estimation
            performance_lb_percentile: estimate the <>-percentile performance as perf. constraint
            success_rate_condition: desired expected success rate as perf. constraint, given 
                                    performance_lower_bound for success condition
                                    (for the sake of implementation, this value replaces
                                    the performance_lower_bound, which is in turn saved as a task-solved
                                    condition into this variable)
            prior_constraint: constraint density around prior point to be atleast equal to uniform density
            force_success_with_returns: force using returns for computing success rate, instead of custom
                                        metric defined in the env (env.success_metric)
            init_beta_param : float, initial value for beta distribution parameters a and b.
                              used for constructing the sigmoid.
            beta_param_bounds: (), set sigmoid bounds to this tuple
        """
        self.training_subrtn = training_subrtn

        self.performance_lower_bound = performance_lower_bound
        self.kl_upper_bound = kl_upper_bound
        
        self.budget = budget
        self.init_budget = budget
        self.max_training_steps = max_training_steps
        self.min_training_steps = 100
        self.stopAtRewardThreshold = stopAtRewardThreshold
        self.reward_threshold_margin = reward_threshold_margin
        self.training_subrtn_kwargs = training_subrtn_kwargs
        self.min_dynamics_samples = min_dynamics_samples
        self.max_dynamics_samples = max_dynamics_samples
        self.reset_agent = reset_agent
        self.train_until_performance_lb = train_until_performance_lb
        self.hard_performance_constraint = hard_performance_constraint
        self.robust_estimate = robust_estimate
        self.alpha_ci = alpha_ci
        self.performance_lb_percentile = performance_lb_percentile
        self.success_rate_condition = success_rate_condition
        self.prior_constraint = prior_constraint
        self.n_iter_skipped = 0
        self.train_until_done = False 
        self.bootstrap_values = bootstrap_values
        self.test_episodes = test_episodes
        self.force_success_with_returns = force_success_with_returns

        self.init_distr = deepcopy(init_distr)
        self.current_distr = init_distr
        self.target_distr = target_distr

        # Lower the bounds if starting from a uniform distribution
        if beta_param_bounds is None:
            self.min_bound = 1. if self.train_until_performance_lb else 0.8
            self.max_bound = init_beta_param + 10  # a bit larger than the initial value
        else:
            margin = 0.1*(beta_param_bounds[1] - beta_param_bounds[0])
            self.min_bound = max(0, beta_param_bounds[0] - margin)
            self.max_bound = beta_param_bounds[1] + margin

            # Make sure bounds are at least these ones.
            self.min_bound = min(1, self.min_bound)
            self.max_bound = max(init_beta_param + 10, self.max_bound)

        self.current_iter = 0
        self.distr_history = []
        self.previous_policy = None
        # self.best_policy = None
        self.best_policy_return = -np.inf
        self.best_policy_succ_rate = -1
        self.best_distr = None
        self.best_iter = None

        self.verbose = verbose

        if self.success_rate_condition is not None:
            assert self.success_rate_condition >= 0. and self.success_rate_condition <= 1., 'Desired expected success rate should be in [0, 1]'
            assert not self.stopAtRewardThreshold, 'Sanity stop. Reward_threshold passed to the training subroutine would now be expressed ' \
                                                   'as success_rate, which is not correct. You may fix this if of interest.'
            # Change variables just for the sake of ease of implementation, so that
            # I can keep the performance_lower_bound the same across different
            # experimental settings
            task_solved_condition = self.performance_lower_bound
            self.performance_lower_bound = self.success_rate_condition
            self.success_rate_condition = task_solved_condition

        self.prior_point = self._get_prior_point(init_distr).reshape(1, -1)  # reshape for batch dimension
        self.uniform_density_at_prior = 0.

    def is_there_budget_for_iter(self):
        return self.budget > self.min_training_steps

    def dummy_doraemon_update(self, kl_step=None):
        curr_step_kl = 0. if kl_step is None else kl_step
        curr_kl_from_target =  self.current_distr.kl_divergence(self.target_distr)
        curr_entropy = self.current_distr.entropy().item()
        wandb.log({"kl_step": curr_step_kl, 'timestep': self.current_timestep})
        wandb.log({"kl_from_target": curr_kl_from_target, 'timestep': self.current_timestep})
        wandb.log({"entropy": curr_entropy, 'timestep': self.current_timestep})
        wandb.run.summary[f'distr_iter_{self.current_iter}'] = self.current_distr.to_string()
        if self.verbose >= 1:
            print(f'KL step before vs after update:', curr_step_kl)
            print(f'KL from target after update:', curr_kl_from_target)
            print(f'Entropy after update:', curr_entropy)


        self.distr_history.append(deepcopy(self.current_distr))
        self.current_iter += 1
        self.training_subrtn.reset_buffer()
        return

    def step(self,
             ckpt_dir: str = None):
        """
        Perform a step of DORAEMON. This includes the training subroutine
        and updating the DR distribution accordingly.

            ckpt_dir : path for saving checkpoint before opt. problem 
        """
        if not self.is_there_budget_for_iter():
            print('Budget has finished. No .step() is allowed.')
            return False

        if self.verbose >= 1:
            print(f'\n=== DORAEMON Step {self.current_iter}')
            print(f'current distribution params: {self.current_distr.get_stacked_params()}')

        if len(self.distr_history) == 0:
            # first iteration
            self.distr_history.append(deepcopy(self.current_distr))
            print('Init KL from target:', self.current_distr.kl_divergence(self.target_distr).item())
            print('Init entropy:', self.current_distr.entropy().item())
            wandb.log({"entropy": self.current_distr.entropy().item(), 'timestep': 0})
            print('Maximum entropy achievable:', self.target_distr.entropy().item())

        """
            1. Train policy theta_i+1 with domain randomization phi_i
        """
        start = time.time()
        training_budget = min(self.budget, self.max_training_steps)
        mean_reward, std_reward, policy, agent, eff_timesteps = self.training_subrtn.train(domainRandDistribution=self.current_distr,
                                                                                           doraemon_iter=self.current_iter,
                                                                                           performance_threshold=self.performance_lower_bound + self.reward_threshold_margin,
                                                                                           stopAtRewardThreshold=self.stopAtRewardThreshold,
                                                                                           max_training_steps=training_budget,
                                                                                           reset_agent=self.reset_agent,
                                                                                           **self.training_subrtn_kwargs)
        self.budget -= eff_timesteps  # decrease budget
        wandb.log({"budget": self.budget, 'timestep': self.current_timestep})
        wandb.log({"train_mean_reward": mean_reward, "timestep": self.current_timestep})
        wandb.log({"train_stdev_reward": std_reward, "timestep": self.current_timestep})
        if self.verbose >= 1:
            print(f'Mean return at iter {self.current_iter} (ts: {self.current_timestep}): {mean_reward} +- {std_reward}')
        if self.verbose >= 2:
            print(f"policy training time (s): {round(time.time() - start, 2)}")
            print(f'Budget stats: {eff_timesteps} (used in last iteration) | {self.budget} (remaining)')

        start = time.time()
        dynamics_params, original_values = self._get_buffer()  # retrieve training statistics (dynamics, returns)
        values = torch.tensor(original_values)

        succ_metrics = self._get_succ_metric_buffer()  # retrieve custom metric for computing success collected during training (optional)

        # Replace returns with the custom metric, if defined
        if len(succ_metrics) > 0 and not self.force_success_with_returns:
            values = torch.tensor(succ_metrics)

        train_success_rate = torch.tensor(values >= self.task_solved_threshold, dtype=torch.float32).mean()

        wandb.log({"est_train_mean_reward": original_values.mean(), "timestep": self.current_timestep})
        wandb.log({"est_train_median_reward": np.median(original_values), "timestep": self.current_timestep})
        wandb.log({"est_train_success_rate": train_success_rate, "timestep": self.current_timestep})
        if len(succ_metrics) > 0:
            wandb.log({"est_train_mean_succ_metric": succ_metrics.mean(), "timestep": self.current_timestep})
            wandb.log({"est_train_median_succ_metric": np.median(succ_metrics), "timestep": self.current_timestep})
        if self.verbose >= 2:
            print(f"Retrieve training stats (dynamics, returns) (s): {round(time.time() - start, 2)}")
            print(f'Number of dynamics samples from buffer: {len(dynamics_params)}')
            print(f'Estimated mean return at iter {self.current_iter}: {original_values.mean()}')
            print(f'Estimated median return at iter {self.current_iter}: {np.median(original_values)}')
            if len(succ_metrics) > 0:
                print(f'Estimated mean succ metric at iter {self.current_iter}: {succ_metrics.mean()}')
                print(f'Estimated median succ metric at iter {self.current_iter}: {np.median(succ_metrics)}')
            print(f'Estimated train succ rate at iter {self.current_iter}: {train_success_rate}')

        start = time.time()
        self.test_on_target_distr(policy, ckpt_dir)  # test current policy on max entropy distribution
        if self.verbose >= 2:
            print(f"policy eval time on target distr (s): {round(time.time() - start, 2)}")
        self.previous_policy = deepcopy(policy)

        # Save checkpoint before opt. problem
        if ckpt_dir is not None:
            additional_keys = {'mean_reward': mean_reward, 'std_reward': std_reward, 'succ_metrics': succ_metrics, 'values': original_values, 'dynamics': dynamics_params, 'best_policy_succ_rate': self.best_policy_succ_rate, 'best_policy_return': self.best_policy_return, 'policyfilename': 'overall_best.pth'}
            self._save_checkpoint(save_dir=ckpt_dir, additional_keys=additional_keys)


        """
            2. Optimize KL(phi_i+1 || phi_target) s.t. J(phi_i+1) > performance_bound & KL(phi_i+1 || phi_i) < KL_bound
        """
        constraints = []

        def prior_constraint_fn(x_opt):
            """Compute log-density of current distribution at prior point"""
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)

            prior_density = proposed_distr.pdf(self.prior_point, log=True, standardize=True)
            return prior_density.detach().numpy()[0]

        def prior_constraint_fn_prime(x_opt):
            """Compute the derivative of log-density of current distribution at prior point"""
            x_opt = torch.tensor(x_opt, requires_grad=True)
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)

            prior_density = proposed_distr.pdf(self.prior_point, log=True, requires_grad=True, standardize=True, to_params=x)[0]
            grads = torch.autograd.grad(prior_density, x_opt)
            return np.concatenate([g.detach().numpy() for g in grads])

        if self.prior_constraint:
            constraints.append(
                NonlinearConstraint(
                    fun=prior_constraint_fn,
                    lb=self.uniform_density_at_prior,
                    ub=np.inf,
                    jac=prior_constraint_fn_prime,
                    keep_feasible=False
                )
            )

        def kl_constraint_fn(x_opt):
            """Compute KL-divergence between current and proposed distribution."""
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)
            kl_divergence = self.current_distr.kl_divergence(proposed_distr)
            return kl_divergence.detach().numpy() 

        def kl_constraint_fn_prime(x_opt):
            """Compute the derivative for the KL-divergence (used for scipy optimizer)."""
            x_opt = torch.tensor(x_opt, requires_grad=True)
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)
            kl_divergence, p_params, q_params  = self.current_distr.kl_divergence(proposed_distr, requires_grad=True, q_params=x)
            grads = torch.autograd.grad(kl_divergence, x_opt)
            return np.concatenate([g.detach().numpy() for g in grads])

        constraints.append(
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                ub=self.kl_upper_bound,
                jac=kl_constraint_fn_prime,
                keep_feasible=self.hard_performance_constraint
            )
        )

        def performance_constraint_fn(x_opt, force_robust=None):
            """Compute the expected performance under the proposed distribution."""
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)

            importance_sampling = torch.exp(proposed_distr.pdf(dynamics_params, log=True, standardize=True) - self.current_distr.pdf(dynamics_params, log=True, standardize=True))

            if self.success_rate_condition is not None:
                # Perf. constraint as expected success rate
                perf_values = torch.tensor(values.detach() >= self.success_rate_condition, dtype=torch.float64)
            else:
                # Perf. constraint as expected return
                perf_values = values

            performance = torch.mean(importance_sampling * perf_values)

            if (force_robust == True or self.robust_estimate) and (force_robust != False):
                # alpha-lower confidence bound used as a constraint
                N = dynamics_params.shape[0]
                var_is_est = self.variance_IS_estimator(dynamics_params, perf_values, proposed_distr, self.current_distr, standardize=True)  # compute est. variance of ImportanceSampling estimator

                if self.performance_lb_percentile is None:
                    # Constraint on lower confidence bound for reward mean
                    lcb, ucb = self._get_ci(mean=performance, stdev=torch.sqrt(var_is_est), N=N, alpha=self.alpha_ci)  # compute CI
                    # print(f'performance estimate under proposed distr: {performance} +- {torch.sqrt(var_is_est)} [{lcb}, {ucb}] ----- {x_opt}')
                    return lcb.detach().numpy()
                else:
                    # Constraint on percentile of rewards
                    # assuming Gaussian rewards ~ N (performance, torch.sqrt(var_is_est*N))
                    dist = Normal(performance, torch.sqrt(var_is_est*N))
                    percentile_reward = dist.icdf(torch.tensor([self.performance_lb_percentile]))
                    # print(f'performance percentile estimate under proposed distr: {round(self.performance_lb_percentile*100, 0)}-th percentile {percentile_reward} [mean: {performance}, stdev: {torch.sqrt(var_is_est*N)}]')
                    return percentile_reward.detach().numpy()

            else:
                return performance.detach().numpy()

        def performance_constraint_fn_prime(x_opt):
            """Compute the derivative for the performance-constraint (used for scipy optimizer)."""
            x_opt = torch.tensor(x_opt, requires_grad=True)
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)

            proposed_distr_log_prob, q_params = proposed_distr.pdf(dynamics_params, log=True, requires_grad=True, standardize=True, to_params=x)
            importance_sampling = torch.exp(proposed_distr_log_prob - self.current_distr.pdf(dynamics_params, log=True, standardize=True))

            if self.success_rate_condition is not None:
                # Perf. constraint as expected success rate
                perf_values = torch.tensor(values.detach() >= self.success_rate_condition, dtype=torch.float64)
            else:
                # Perf. constraint as expected return
                perf_values = values
            
            performance = torch.mean(importance_sampling * perf_values)

            if self.robust_estimate:
                # alpha-lower confidence bound used as a constraint 
                N = dynamics_params.shape[0]
                var_is_est = self.variance_IS_estimator(dynamics_params, perf_values, proposed_distr, self.current_distr, standardize=True, requires_grad=True, to_params=x)  # compute est. variance of ImportanceSampling estimator

                if self.performance_lb_percentile is None:
                    # Constraint on lower confidence bound for reward mean
                    lcb, ucb = self._get_ci(mean=performance, stdev=torch.sqrt(var_is_est), N=N, alpha=self.alpha_ci)  # compute CI
                    # print(f'performance estimate under proposed distr: {performance} +- {torch.sqrt(var_is_est)} [{lcb}, {ucb}] ----- {x_opt}')
                    grads = torch.autograd.grad(lcb, x_opt)
                    return np.concatenate([g.detach().numpy() for g in grads])
                else:
                    # Constraint on percentile of rewards
                    # assuming Gaussian rewards ~ N (performance, torch.sqrt(var_is_est*N))
                    dist = Normal(performance, torch.sqrt(var_is_est*N))
                    percentile_reward = dist.icdf(torch.tensor([self.performance_lb_percentile]))
                    # print(f'performance percentile estimate under proposed distr: {round(self.performance_lb_percentile*100, 0)}-th percentile {percentile_reward} [mean: {performance}, stdev: {torch.sqrt(var_is_est*N)}]')
                    grads = torch.autograd.grad(percentile_reward, x_opt)
                    return np.concatenate([g.detach().numpy() for g in grads])
            else:
                grads = torch.autograd.grad(performance, x_opt)
                return np.concatenate([g.detach().numpy() for g in grads])

        constraints.append(
            NonlinearConstraint(
                fun=performance_constraint_fn,
                lb=self.performance_lower_bound-1e-4,  # scipy would still complain if x0 is very close to the boundary
                ub=np.inf,
                jac=performance_constraint_fn_prime,
                keep_feasible=self.hard_performance_constraint
            )
        )

        def objective_fn(x_opt):
            """Minimize KL-divergence between the current and the target distribution,
                s.t. previously defined constraints."""
            x_opt = torch.tensor(x_opt, requires_grad=True)
            x = DomainRandDistribution.sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = DomainRandDistribution.beta_from_stacked(stacked_bounds=self.current_distr.get_stacked_bounds(),
                                                                      stacked_params=x)

            kl_divergence, p_params, q_params = proposed_distr.kl_divergence(self.target_distr, requires_grad=True, p_params=x)
            grads = torch.autograd.grad(kl_divergence, x_opt)

            return (
                kl_divergence.detach().numpy(),
                np.concatenate([g.detach().numpy() for g in grads]),
            )


        x0 = self.current_distr.get_stacked_params()
        x0_opt = DomainRandDistribution.inv_sigmoid(x0, self.min_bound, self.max_bound)


        """
            Skip DORAEMON optimization at the beginning until performance_lower_bound is reached 
        """
        if self.train_until_performance_lb and not self.train_until_done:
            if performance_constraint_fn(x0_opt) < self.performance_lower_bound:
                # Skip DORAEMON update
                print(f'--- DORAEMON iter {self.current_iter} skipped as performance lower bound has not been reached yet. Mean reward {performance_constraint_fn(x0_opt)} < {self.performance_lower_bound}')
                wandb.log({"update": -1, 'timestep': self.current_timestep})
                self.dummy_doraemon_update()
                return
            else:
                # Skip iterations only once, until you reach it the first time
                self.train_until_done = True
                self.n_iter_skipped = 0


        """
            Start from a feasible distribution within the trust region Kl(p||.) < eps
        """
        if self.hard_performance_constraint:
            if performance_constraint_fn(x0_opt) < self.performance_lower_bound:
                # Performance constraint not satisfied. Find a different initial distribution within the current trust region
                if self.verbose >= 2:
                    print(f'Solving the inverted problem as current distr is not feasible. Perf. constraint value {performance_constraint_fn(x0_opt)} < {self.performance_lower_bound}')
                max_perf_x0_opt, curr_step_kl, success = self.get_feasible_starting_distr(x0_opt=x0_opt,
                                                                                          obj_fn=performance_constraint_fn,
                                                                                          obj_fn_prime=performance_constraint_fn_prime,
                                                                                          kl_constraint_fn=kl_constraint_fn,
                                                                                          kl_constraint_fn_prime=kl_constraint_fn_prime)
                if success:
                    if performance_constraint_fn(max_perf_x0_opt) >= self.performance_lower_bound:
                        # Feasible distribution found, Go on with this new starting distribution
                        x0_opt = max_perf_x0_opt
                        x0 = DomainRandDistribution.sigmoid(x0_opt, self.min_bound, self.max_bound)
    
                        wandb.log({"update": 1, 'timestep': self.current_timestep})
                        if self.verbose >= 2:
                            print(f'New feasible start distribution found: {x0.detach().numpy()}. ' \
                                  f'Est reward constraint value: {performance_constraint_fn(max_perf_x0_opt)} >= {self.performance_lower_bound}')

                    else:
                        # No feasible distribution within the trust region has been found
                        # Keep training with the max performance distribution within the trust region
                        new_x = DomainRandDistribution.sigmoid(max_perf_x0_opt, self.min_bound, self.max_bound)
                        self.current_distr.update_parameters(new_x)
                        wandb.log({"update": -2, 'timestep': self.current_timestep})
                        print(f'No distribution within the trust region satisfies the performance_constraint. ' \
                              f'Keep training with the max performant distribution in the trust region: {new_x.detach().numpy()} ' \
                              f'Est reward constraint value: {performance_constraint_fn(max_perf_x0_opt)} < {self.performance_lower_bound}')
                        self.dummy_doraemon_update(kl_step=curr_step_kl)
                        return

                else:
                    # Inverse opt. problem had an unexpected error
                    print('Warning! inverted optimization problem NOT successful.')
                    wandb.log({"update": -3, 'timestep': self.current_timestep})
                    self.dummy_doraemon_update()
                    return
            else:
                # The current policy already satisfies the performance constraint. Go on
                wandb.log({"update": 0, 'timestep': self.current_timestep})
            

        if self.verbose >= 2:
            print("Performing DORAEMON update.")

        start = time.time()
        result = minimize(
            objective_fn,
            x0_opt,
            method="trust-constr",
            jac=True,
            constraints=constraints,
            options={"gtol": 1e-4, "xtol": 1e-6},
        )
        if self.verbose >= 2:
            print(f"scipy optimization time (s): {round(time.time() - start, 2)}")

        new_x_opt = result.x

        # Check validity of new optimum found
        if not result.success:
            print('Warning! optimization NOT successful.')
            # If optimization process was not a success
            old_f = objective_fn(x0_opt)[0]
            constraints_satisfied = [const.lb <= const.fun(result.x) <= const.ub for const in constraints]

            if self.verbose >= 1:
                print('--- constraints satisfied:', constraints_satisfied)

            if not (all(constraints_satisfied) and result.fun < old_f):  # keep old parameters if update was unsuccessful
                print(f"Warning! Update effectively unsuccessful, keeping old values parameters.")
                new_x_opt = x0_opt

        new_x = DomainRandDistribution.sigmoid(new_x_opt, self.min_bound, self.max_bound)

        if self.verbose >= 1:
            print(f'New best params: {new_x}')
            print(f'Est mean performance in new distribution: {performance_constraint_fn(new_x_opt, force_robust=False)}{" > "+str(self.performance_lower_bound) if not self.robust_estimate else ""}')
            print(f'Est robust constraint performance in new distribution: {performance_constraint_fn(new_x_opt, force_robust=True)}{" > "+str(self.performance_lower_bound) if self.robust_estimate else ""}')
            print(f'Log-density at prior point:', prior_constraint_fn(new_x_opt))
        wandb.log({"est_NEW_mean_performance": performance_constraint_fn(new_x_opt, force_robust=False), "timestep": self.current_timestep})
        wandb.log({"est_NEW_robust_constraint_performance": performance_constraint_fn(new_x_opt, force_robust=True), "timestep": self.current_timestep})
        wandb.log({"density_prior": prior_constraint_fn(new_x_opt), "timestep": self.current_timestep})

        curr_step_kl = kl_constraint_fn(new_x_opt)  # to compute before the distribution is updated

        ### Update domain randomization distribution
        self.current_distr.update_parameters(new_x)
        wandb.run.summary[f'distr_iter_{self.current_iter}'] = self.current_distr.to_string()

        curr_kl_from_target = objective_fn(new_x_opt)[0]
        curr_entropy = self.current_distr.entropy().item()
        wandb.log({"kl_step": curr_step_kl, 'timestep': self.current_timestep})
        wandb.log({"kl_from_target": curr_kl_from_target, 'timestep': self.current_timestep})
        wandb.log({"entropy": curr_entropy, 'timestep': self.current_timestep})
        if self.verbose >= 1:
            print(f'KL step before vs after update:', curr_step_kl)
            print(f'KL from target after update:', curr_kl_from_target)
            print(f'Entropy after update:', curr_entropy)

        self.distr_history.append(deepcopy(self.current_distr))
        self.current_iter += 1
        self.training_subrtn.reset_buffer()

        return


    def _get_ci(self, mean, stdev, N, alpha):
        """Compute alpha-confidence interval"""
        t_score = float(st.t.ppf((1 + alpha) / 2., N ))  # N degrees of freedom instead of (N-1), because I'm using the ImpSapling estimation.
        ci = t_score * stdev / (N**0.5)
        return mean - ci, mean + ci


    def variance_IS_estimator(self, x, f_x, p, q, standardize=True, requires_grad=False, to_params=None):
        """Compute estimate of variance of the
            Importance Sampling (IS) estimator statistics.
            IS = E_q[X * p(X) / q(X)]
            IS_est = 1/N sum ( x[i] * p(x[i] / q(x[i]) )
            X[i] are I.I.D, therefore:
            Var(IS_est) = 1/N (E_q[X^2 * p^2(X) / q^2(X)] - IS^2 )

            params:
                x : list of samples from q
                f_x : mapped samples according to some f(x) whose expected value
                      wants to be estimated w.r.t. p through IS
                p : distribution of interest for computing E_p[X]
                q : sampling distribution of x
                requires_grad: bool
                               if set, track gradient w.r.t. p
                to_params: torch distribution parameters to use for gradient propagation
        """
        N = x.shape[0]

        if requires_grad:
            p_log_prob, p_params = p.pdf(x, log=True, standardize=standardize, requires_grad=True, to_params=to_params)
        else:
            p_log_prob = p.pdf(x, log=True, standardize=standardize)

        squared_importance_sampling_ratio = torch.exp(2*p_log_prob - 2*q.pdf(x, log=True, standardize=standardize))
        second_moment_est = torch.mean( torch.square(f_x) * squared_importance_sampling_ratio ) 

        importance_sampling_ratio = torch.exp(p_log_prob - q.pdf(x, log=True, standardize=standardize)) 
        squared_first_moment_est = torch.mean( f_x * importance_sampling_ratio)**2

        var_is_est = 1/N * ( second_moment_est - squared_first_moment_est )
        return var_is_est

    def get_feasible_starting_distr(self, x0_opt, obj_fn, obj_fn_prime, kl_constraint_fn, kl_constraint_fn_prime):
        """
            Solves the inverted problem
            max J(phi_i+1) s.t. KL(phi_i+1 || phi_i) < eps
            to find an initial feasible distribution
        """
        def negative_obj_fn_with_grad(x_opt):
            return (
                -1 * obj_fn(x_opt),
                -1 * obj_fn_prime(x_opt)
            )

        constraints = []
        constraints.append(
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                # make sure you don't get too close the original trust region bound for numerical stability during subsequent DORAEMON update
                ub=self.kl_upper_bound-1e-5,
                jac=kl_constraint_fn_prime,
                keep_feasible=True,
            )
        )

        start = time.time()
        result = minimize(
            negative_obj_fn_with_grad,  # maximization
            x0_opt,
            method="trust-constr",
            jac=True,
            constraints=constraints,
            options={"gtol": 1e-4, "xtol": 1e-6},
        )
        if self.verbose >= 2:
            print(f"scipy inverted problem optimization time (s): {round(time.time() - start, 2)}")

        if not result.success:
            return None, None, False
        else:
            feasible_x0_opt = result.x
            curr_step_kl = kl_constraint_fn(feasible_x0_opt)
            return feasible_x0_opt, curr_step_kl, True


    def reset(self):
        self.current_iter = 0
        self.current_distr = self.init_distr
        self.distr_history = []
        self.previous_policy = None
        return

    def test_on_target_distr(self, policy, ckpt_dir=None):
        """Evaluate policy on target distribution and
            keep track of best overall policy found

            ckpt_dir: save best in this dir if provided
        """
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': True})
        self.training_subrtn.eval_env.env_method('reset_buffer')
        target_mean_reward, target_std_reward = self.training_subrtn.eval(policy, self.target_distr, eval_episodes=self.test_episodes)
        
        returns = np.array(self._flatten(self.training_subrtn.eval_env.env_method('get_buffer')))  # retrieve tracked returns
        succ_metrics = np.array(self._flatten(self.training_subrtn.eval_env.env_method('get_succ_metric_buffer')))  # retrieve custom metric for measuring success, and flatten values

        if len(succ_metrics) == 0 or self.force_success_with_returns:
            # Use returns for measuring success
            target_success_rate = torch.tensor(returns >= self.task_solved_threshold, dtype=torch.float32).mean()
        else:
            # Use custom metric for measuring success
            target_success_rate = torch.tensor(succ_metrics >= self.task_solved_threshold, dtype=torch.float32).mean()

        # target_success_rate = torch.tensor(returns >= self.task_solved_threshold, dtype=torch.float32).mean()
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': False})
        self.training_subrtn.eval_env.env_method('reset_buffer')

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

    def _handle_min_max_dynamics_samples(self, dynamics_params, distr, min_samples, max_samples):
        """
            Make sure the number of dynamics sampled
            returned by the RL subroutine are at least
            as many as min_samples and at most max_samples.

            dynamics_params: list of dynamics parameters sampled
            distr: DomainRandDistribution to sample from in case too
                   few parameters are given
            min_samples: int, minimum number of samples
            max_samples: int, maximum number of samples
        """
        if len(dynamics_params) < min_samples:  # up-sample more parameters
            return np.concatenate((dynamics_params, distr.sample(n_samples=(min_samples-len(dynamics_params)))))

        elif len(dynamics_params) > max_samples:  # down-sample parameters
            selected_rows = np.random.choice(len(dynamics_params), size=max_samples, replace=False)
            return dynamics_params[selected_rows, :]

        return dynamics_params

    def _get_buffer(self):
        return self.training_subrtn.get_buffer()

    def _get_succ_metric_buffer(self):
        return self.training_subrtn.get_succ_metric_buffer()

    def _save_checkpoint(self, save_dir, additional_keys: Dict = {}):
        """Save DORAEMON checkpoint"""
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

    def _get_prior_point(self, distr):
        """Returns point at the center of the beta distribution"""
        bounds = distr.get_stacked_bounds()
        prior_point = []

        for i in range(len(bounds)//2):
            prior_point.append(  0.5*bounds[i*2] + 0.5*bounds[i*2 + 1] )

        return np.array(prior_point)

    def _flatten(self, multi_list):
        """Flatten a list of lists with potentially
        different lenghts into a 1D np array"""
        flat_list = [] 
        for single_list in multi_list:
            flat_list += single_list
        return np.array(flat_list, dtype=np.float64)

    @property
    def current_timestep(self):
        """
            return the current number of training timesteps consumed
        """
        return self.init_budget - self.budget

    @property
    def task_solved_threshold(self):
        """
            return the task-solved threshold value.
            Such value is held in a different variable
            depending on the setting.
        """
        return self.performance_lower_bound if self.success_rate_condition is None else self.success_rate_condition