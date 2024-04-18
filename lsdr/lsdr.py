"""
    Standalone custom implementation of LSDR
    (https://ieeexplore.ieee.org/document/9341019)
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import sys
import pdb
from copy import deepcopy
import time
import itertools
from multiprocessing import Pool

import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from scipy.stats import multivariate_normal
# from scipy import integrate

from utils.utils import *
from utils.gym_utils import *
from policy.policy import Policy


class TrainingSubRtn():
    """Training subroutine wrapper
        
        Used by LSDR class to handle
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
              training_steps,
              before_final_eval_cb=None,
              after_final_eval_cb=None,
              reset_agent=False):
        """Trains a policy until reward
            threshold is reached, or for a maximum
            number of steps.
        """
        self.env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get())
        self.env.set_dr_training(True)

        self.env.reset()

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

        mean_reward, std_reward, best_policy, which_one = self.agent.train(timesteps=training_steps,
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

    def eval(self, policy, domainRandDistribution, env, eval_episodes=None):
        """Evaluate policy with given DR distribution
            on self.env"""
        env.set_dr_distribution(dr_type=domainRandDistribution.get_dr_type(), distr=domainRandDistribution.get_params())
        env.set_dr_training(True)
        agent = Policy(algo=self.algo,
                       env=env,
                       device=self.device,
                       seed=self.seed,
                       actor_obs_mask=self.actor_obs_mask,
                       critic_obs_mask=self.critic_obs_mask)
        agent.load_state_dict(policy)

        mean_reward, std_reward = agent.eval(n_eval_episodes=(self.n_eval_episodes if eval_episodes is None else eval_episodes))
        env.set_dr_training(False)

        return mean_reward, std_reward



class DomainRandDistribution():
    """Handles Domain Randomization distributions"""

    def __init__(self,
                 distr):
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

    def sample(self, n_samples=1):
        samples = torch.zeros((n_samples, self.ndims))
        for i in range(self.ndims):
            samples[:, i] = self.to_distr[i].sample(sample_shape=(n_samples,))
        return samples

    def pdf(self, x, log=False, requires_grad=False):
        return torch.zeros((x.shape[0],))

        # assert requires_grad

        # if log:
        #     return self.to_distr.log_prob(x)
        # else:
        #     raise NotImplementedError

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


class MultivariateGaussian(DomainRandDistribution):
    """Uniform Domain Randomization distribution"""

    def __init__(self,
                 distr: Dict,
                 now: int):
        DomainRandDistribution.__init__(self,
                                        distr)
        """
            Multivariate Gaussian distribution

            distr: Dict
                   expected as {'mean': ndarray(N), 'cov': ndarray(N,N)}
            now: number of cpu cores for parallelizing entropy computation
        """

        self.dr_type = 'multivariateGaussian'
        self.now = now
        return

    def set(self, distr):   
        """Sets distribution"""
        assert 'mean' in distr and 'cov' in distr
        
        self.distr = distr.copy()
        self.mean = distr['mean']
        assert isinstance(self.mean, np.ndarray)

        self.ndims = len(self.mean)
        self.cov = distr['cov']
        assert isinstance(self.cov, np.ndarray)

        # Bounds for denormalization
        self.low, self.high = None, None
        if 'low' in distr:
            assert 'high' in distr
            # Assume samples to be normalized from [low, high] to [0 , 1]
            # Save bounds to potentially denormalize samples
            self.low = np.array(distr['low'])
            self.high = np.array(distr['high'])

        # Transform sigma into L
        self.to_mean = torch.tensor(self.mean)

        to_cov = torch.tensor(self.cov)
        L = torch.linalg.cholesky(to_cov).double()
        D = L.size(0)
        mask = torch.ones(D, D).tril(-1).bool()  # all elements strictly below the main diagonal
        self.L_low_diag = torch.zeros(D, D).double()

        self.L_off_diag = torch.tensor(L[mask])  # off-diagonal elements of the lower-side
        self.log_diag_L = torch.log(L.diag())  # log-transformed elements of diagional

        self.to_mean.requires_grad_(True)     # optimize means
        self.L_off_diag.requires_grad_(True)  # optimize lower off-diagonal elements of L
        self.log_diag_L.requires_grad_(True)  # optimize log-diagonal elements of L

        self.L_low_diag[mask] = self.L_off_diag
        self.to_L = self.L_low_diag + torch.eye(D)*torch.exp(self.log_diag_L)  # propagates gradients correctly, instead of using L
        self.L = self.to_L.clone().detach()

        self.to_distr = MultivariateNormal(loc=self.to_mean,
                                           covariance_matrix=None,
                                           precision_matrix=None,
                                           scale_tril=self.to_L)


        self.to_params = [self.to_mean, self.L_off_diag, self.log_diag_L]

    def sample(self, n_samples=1, truncated=False):
        if not truncated:
            return self.to_distr.sample(sample_shape=(n_samples,))
        else:
            # Normalized Truncated Normal distribution in [0, 1]
            n_valids = 0
            non_valid_mask = torch.ones((n_samples)).bool()
            samples = torch.zeros((n_samples, self.ndims)).double()

            low, high = torch.zeros((self.ndims)), torch.ones((self.ndims))
            n_iters = 0
            while n_valids < n_samples:
                samples[non_valid_mask] = self.to_distr.sample(sample_shape=(non_valid_mask.int().sum(),))

                mask_low = torch.greater(samples, low.view(1,-1))
                mask_high = torch.less(samples, high.view(1,-1))
                per_sample_mask_with_dim = torch.cat([mask_low, mask_high], dim=-1) # (n_samples, 2*ndims)
                per_sample_mask = torch.all(per_sample_mask_with_dim, dim=-1)
                non_valid_mask = ~per_sample_mask

                n_valids = per_sample_mask.int().sum()
                n_iters += 1

            if n_iters >= 10:
                print('WARNING! Sampling through the truncated normal took {n_iters} >= 10 iterations for resampling.')
            
            return samples

    def pdf(self, x, log=False, requires_grad=False):
        assert requires_grad

        if log:
            return self.to_distr.log_prob(x)
        else:
            raise NotImplementedError

    def update_distribution(self):
        D = self.L.size(0)
        mask = torch.ones(D, D).tril(-1).bool()
        self.L_low_diag = torch.zeros(D, D).double()
        self.L_low_diag[mask] = self.L_off_diag
        self.to_L = self.L_low_diag + torch.eye(D)*torch.exp(self.log_diag_L)

        self.to_distr = MultivariateNormal(loc=self.to_mean,
                                           covariance_matrix=None,
                                           precision_matrix=None,
                                           scale_tril=self.to_L)


        self.mean = self.to_mean.clone().detach().numpy()
        self.L = self.to_L.clone().detach()
        self.cov = self.L.mm(self.L.t()).numpy()
        self.distr = {'mean': self.mean, 'cov': self.cov, 'low': self.low, 'high': self.high}

    def normalize_samples(self, samples):
        """Normalize samples into standardized space"""
        if torch.is_tensor(samples):
            samples = samples.detach().numpy()

        if samples.ndim == 1:
            samples.reshape(1, -1)

        return (samples - self.low) / (self.high - self.low)


    def denormalize_samples(self, samples):
        """Denormalize samples back in their true space"""
        if torch.is_tensor(samples):
            samples = samples.detach().numpy()

        if samples.ndim == 1:
            samples.reshape(1, -1)

        return (self.high - self.low) * samples + self.low



    def get_stacked_params(self):
        mean, cov = self.get_params()
        return np.concatenate([mean, cov.ravel()])

    def get_params(self):
        return [self.mean, self.cov]

    def get_to_params(self):
        return self.to_params

    def update_parameters(self, params):
        """Update the current parameters"""
        raise NotImplementedError()
        # distr = deepcopy(self.distr)
        # for i in range(self.ndims):
        #     distr[i]['m'] = params[i*2]
        #     distr[i]['M'] = params[i*2 + 1]

        # self.set(distr=distr)

    def entropy(self, standardize=False):
        """Returns entropy of distribution
        
            NOTE: The entropy of the Multivariate Gaussian is not comparable,
        because resampling happens at training time, hence a truncated
        gaussian is actually used.
        """
        if standardize:
            # Distribution is already assumed to be in standardized space
            # return self.to_distr.entropy()
            raise NotImplementedError("Truncated entropy computation is not implemented in standardized space.")
        else:
            # Y = AX + b => H(Y) = H(X) + log(|det A|)
            # return self.to_distr.entropy() + np.sum(np.log(self.high - self.low))  ### OR np.log(np.prod(self.high - self.low))

            if self.ndims <= 13:
                # Compute truncated gaussian entropy instead
                return self.approx_truncated_entropy()
            else:
                # Beyond 13 dimensions the computation becomes unfeasible.
                # Its complexity is O(2^n).
                # Instead, compute the entropy with the independent diagonal approximation.
                # This is for visualization purposes only
                return self.diagonal_approx_truncated_entropy()

    def diagonal_approx_truncated_entropy(self):
        """Compute entropy by using the independent approximation.
            This is for visualization purposes only anyways. The KL
            for the obj function is still always approximated with samples.

            Formula source: https://en.wikipedia.org/wiki/Truncated_normal_distribution
        """
        diagonal = np.diag(self.cov)  # variances
        mean = self.mean

        a, b = 0, 1  # truncated guassian boundaries

        alpha, beta = torch.tensor((a - mean) / np.sqrt(diagonal)), torch.tensor((b - mean) / np.sqrt(diagonal))

        normal = Normal(0, 1)
        cdf_alpha, cdf_beta = normal.cdf(alpha), normal.cdf(beta)
        pdf_alpha, pdf_beta = torch.exp(normal.log_prob(alpha)), torch.exp(normal.log_prob(beta))

        zeta = cdf_beta - cdf_alpha

        entropy_per_dim = np.log(np.sqrt(2*np.pi*np.e)) + np.log(np.multiply(np.sqrt(diagonal), zeta)) + (alpha*pdf_alpha - beta*pdf_beta) / (2*zeta)

        norm_entropy = entropy_per_dim.sum()  # independence assumption
        entropy = norm_entropy + np.sum(np.log(self.high - self.low))  # entropy in original space: Y = aX + b => H(Y) = H(X) + log(a)
        return entropy

    def approx_truncated_entropy(self):
        """Empirical entropy computation of the truncated
        multivariate gaussian in [self.low, self.high]"""

        # Approximate mass in [0, 1]
        start = time.time()
        norm_distr_mass = self.get_mass_within_bounds()
        print('Mass approximation integral (s):', round(time.time() - start, 2))

        # Approximate entropy through samples
        start = time.time()
        norm_entropy = self.get_approx_cross_entropy() + np.log(norm_distr_mass)
        print('CrossEntropy approximation (s):', round(time.time() - start, 2))

        # Denormalize entropy for value in the original space
        entropy = norm_entropy + np.sum(np.log(self.high - self.low))
        return entropy

    def get_approx_cross_entropy(self, nmc=1000000):
        """Computes cross entropy H_distr(trunc_distr).
            I.e., you sample according to the trunc_distr,
            but use the normal pdf for computing the entropy.        
        """
        z = self.sample(nmc, truncated=True).detach()  # sample with truncated distr
        log_p_distr = self.pdf(z, log=True, requires_grad=True).detach()
        ce = -log_p_distr.mean()
        return ce

    def get_mass_within_bounds(self):
        """Estimate probability mass within distribution bounds.
            Note: the normalized space is considered for this computation.
        """

        ######### Approximate through numerical integration (becomes too slow after 5-6 dimensions)
        # rv = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=False)

        # def distr_pdf(*params):
        #     x  = np.array(params[:-1])  # sample
        #     distr = params[-1]  # scipy distribution
        #     return distr.pdf(x)

        # low, high = np.zeros((self.ndims)), np.ones((self.ndims))
        # bounds = np.stack([low, high]).T  # bounds as [[x_low, x_high], [y_low, y_high], ...]

        # result, error = integrate.nquad(distr_pdf, bounds, args=[rv], opts={'epsabs': 100, 'epsrel': 100})


        ######### Use CDF to compute it with formula at:
        ######### https://math.stackexchange.com/questions/106835/generalization-to-n-dimensions-of-distribution-function-evaluation-over-an-hyp
        rv = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=False)

        hyperrectangle = [[0, 1] for _ in range(self.ndims)]
        all_points = list(itertools.product(*hyperrectangle))  # 2^n vectors to compute the CDF in

        
        if self.ndims < 5:
            #### Without parallelization (good if self.ndims < 10)
            mass = 0
            for point in all_points:
                nc = self.ndims - np.sum(point)
                mass += rv.cdf(np.array(point))*((-1)**nc)

        elif self.ndims <= 13:
            #### With parallelization
            now = max(5, self.now)  # number of cpu cores

            # Split the 2^self.ndims points among now CPU cores
            fun_args = []
            n_points = len(all_points)
            for i in range(now):
                low = i*(n_points//now)
                if i == now - 1:
                    # last one
                    high = n_points
                else:
                    high = (i+1)*(n_points//now)

                curr_split = np.array(all_points[ low : high ])
                ncs = (-1)**(self.ndims - np.sum(curr_split, axis=-1))

                fun_args.append([
                                    curr_split,  # split of points
                                    rv,          # distribution
                                    ncs          # factor {-1, 1}
                                ])

            pool = Pool(processes=now)
            res = pool.map(compute_cdf_over_points, fun_args)
            pool.close()
            pool.join()

            mass = np.sum(res)

        else:
            # Beyond 13 dimensions the computation becomes unfeasible.
            # Its complexity is O(2^n)
            raise ValueError('The approximate entropy computation becomes infeasible beyond 13 dimensions.')

        
        return mass

    def print(self):
        print(f'Means: {self.mean}, Cov: {self.cov}')

    def to_string(self):
        string = f'means: {self.mean.round(decimals=4)} | cov: {self.cov.round(decimals=4)}'
        return string


def compute_cdf_over_points(args):
    """Standalone function for computing
    mass of MultivariateNormal in a hyperrectangle.

    Needs to be standalone as Pool must pickle its content,
    and wouldn't work if put inside the class due to PyTorch/pickle errors
    """
    points = args[0]
    distr = args[1]
    ncs = args[2]
    return np.sum(np.multiply(distr.cdf(np.array(points)), ncs))


class LSDR():
    """LSDR implementation

        Original imlementation at:
        https://github.com/melfm/lsdr/

        This implementation shares some pieces of code from the original.
        Credits go to the LSDR authors.

        This version deviates from the original one as (1) it
        uses stable-baselines3 API as RL subroutine, hence it can work
        with any RL algorithm (not just PPO); (2) does not assign minimum
        returns to unfeasible environments (rather, keeps sampling new contexts until
        feasible ones are found); (3) allows policies to be conditioned with different
        information (e.g. historu), instead of always being conditioned with the ground truth dynamics,.
    """
    def __init__(self,
                 training_subrtn: TrainingSubRtn,
                 performance_lower_bound: float,
                 init_distr: DomainRandDistribution,
                 target_distr: DomainRandDistribution,
                 budget: float = 5000000,
                 training_steps: int = 100000,
                 distr_learning_iters: int = 10,
                 test_episodes=100,
                 count_eval_ts_in_budget: bool = True,
                 n_contexts_for_eval: int = 100,
                 alpha: float = 1.0,
                 whiten_performance: bool = True,
                 standardized_performance: bool = False,
                 baseline: float = None,
                 use_kl_regularizer: bool = False,
                 obj_fun_lr : float = 1e-3,
                 force_success_with_returns=False,
                 training_subrtn_kwargs={},
                 verbose=0):
        """
            training_subrtn: handles the RL training subroutine
            performance_lower_bound: performance threshold for determining success rate given returns
                                     or a custom performance metric defined as env.success_metric
            init_distr: starting distribution
            target_distr: target uniform distribution
            budget: total number of env steps allowed.
            training_steps: training steps for the RL training subroutine
            distr_learning_iters : Gradient descent iterations at each distribution update
            count_eval_ts_in_budget : Count monte carlo timesteps for obj. function evaluation
                                      towards the total budget available.
            alpha : float, trade-off parameter in opt. problem
            n_contexts_for_eval : number of sampled contexts for obj. function evaluation
            whiten_performance : standardize returns for a consistent objective function scale
            standardized_performance : in alternative to the exponentially moving average (see whiten_performance)
                                       returns are standardized individually with in-batch statistics
            baseline : fix value to subtract the returns from                                        
            use_kl_regularizer : compute KL divergence empirically, assuming independence
            obj_fun_lr : lr for Adam optimizer for LSDR objective function
            force_success_with_returns: force using returns for computing success rate, instead of custom
                                        metric defined in the env (env.success_metric)
        """
        self.training_subrtn = training_subrtn

        self.performance_lower_bound = performance_lower_bound        
        self.init_budget = budget
        self.budget = budget
        self.training_steps = training_steps
        self.distr_learning_iters = distr_learning_iters
        self.min_training_steps = 100
        self.count_eval_ts_in_budget = count_eval_ts_in_budget
        self.n_contexts_for_eval = n_contexts_for_eval
        self.whiten_performance = whiten_performance
        self.standardized_performance = standardized_performance
        self.baseline = baseline
        self.use_kl_regularizer = use_kl_regularizer
        self.obj_fun_lr = obj_fun_lr 
        self.training_subrtn_kwargs = training_subrtn_kwargs

        self.test_episodes = test_episodes
        self.force_success_with_returns = force_success_with_returns

        self.current_distr = init_distr
        self.target_distr = target_distr

        self.cum_timesteps_for_eval = 0  # keep track of how many timesteps are used for MC evaluation
        self.cum_timesteps_for_training = 0  # keep track of how many timesteps are used for training the policy
        self.target_entropy = self.target_distr.entropy().item()
        self.alpha = alpha
        self.alpha = self.alpha / np.abs(self.target_entropy)  # overwrite alpha so that KL term is rescaled

        self.current_iter = 0
        self.distr_history = []
        self.previous_policy = None
        # self.best_policy = None
        self.best_policy_return = -np.inf
        self.best_policy_succ_rate = -1
        self.best_distr = None
        self.best_iter = None

        # Reward statistics
        self.beta = 0.01
        self.R_mean, self.R_std = 0.0, 1.0

        self.distr_optimizer = torch.optim.Adam(init_distr.get_to_params(), self.obj_fun_lr)
        self.distr_optimizer.zero_grad()

        self.verbose = verbose

        assert not (self.whiten_performance and self.standardized_performance), 'Either one of the two standardization techniques must be chosen.'



    def learn(self,
              ckpt_dir: str = None):
        """
            Run LSDR

            ckpt_dir : path for saving checkpoints
        """
        self.distr_history.append(MultivariateGaussian(self.current_distr.get(), now=self.current_distr.now))
        init_entropy = self.current_distr.entropy().item()
        wandb.log({"entropy": init_entropy, 'timestep': 0})
        print('Init entropy:', init_entropy)
        print('Maximum entropy achievable:', self.target_entropy)


        while self.is_there_budget_for_iter():
            """
                1. Call the training sub routine for a fixed number of timesteps
            """
            start = time.time()
            training_budget = min(self.budget, self.training_steps)
            reward, std_reward, policy, agent, eff_timesteps = self.training_subrtn.train(domainRandDistribution=self.current_distr,
                                                                                          training_steps=training_budget,
                                                                                          before_final_eval_cb=self.before_final_eval_cb,
                                                                                          after_final_eval_cb=self.after_final_eval_cb,
                                                                                          **self.training_subrtn_kwargs)
            self.budget -= eff_timesteps  # decrease remaining budget
            self.cum_timesteps_for_training += eff_timesteps
            wandb.log({"cum_timesteps_for_training": self.cum_timesteps_for_training, "timestep": self.current_timestep})
            
            individual_returns = self._flatten(self.training_subrtn.eval_env.env_method('get_buffer'))  # retrieve returns during final eval
            individual_succmetrics = self._flatten(self.training_subrtn.eval_env.env_method('get_succ_metric_buffer'))  # retrieve custom metric during final eval
            
            # Use custom metric for defining success rate, if defined
            if len(individual_succmetrics) > 0 and not self.force_success_with_returns:
                individual_metrics = individual_succmetrics.copy()
            else:
                individual_metrics = individual_returns.copy()

            train_success_rate = torch.tensor(individual_metrics >= self.performance_lower_bound, dtype=torch.float32).mean()

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
                additional_keys = {'mean_reward': reward,
                                   'std_reward': std_reward,
                                   'best_policy_succ_rate': self.best_policy_succ_rate,
                                   'best_policy_return': self.best_policy_return,
                                   'last_policy_filename': 'overall_best.pth'}
                self._save_checkpoint(save_dir=ckpt_dir, additional_keys=additional_keys)


            """
                2. Update distribution
            """
            ##### TEST MULTIVARIATE GAUSSIAN ################################
            # sample_contexts = self.current_distr.sample(100000)
            # # sample_contexts = np.array([self.training_subrtn.env.env_method('sample_task') for _ in range(5000)]).reshape(5000, 2)
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
            # ax.scatter(sample_contexts[:, 0], sample_contexts[:, 1], s=10, c='blue', alpha=0.2, marker='o')
            # boundaries = self.target_distr.get_params()
            # ax.axvline(x=boundaries[0], color='grey', linestyle='--', alpha=0.5, lw=5)
            # ax.axvline(x=boundaries[1], color='grey', linestyle='--', alpha=0.5, lw=5)
            # ax.axhline(y=boundaries[2], color='grey', linestyle='--', alpha=0.5, lw=5)
            # ax.axhline(y=boundaries[3], color='grey', linestyle='--', alpha=0.5, lw=5)
            # plt.show()
            #################################################################

            start = time.time()
            contexts, returns, timesteps = self.get_monte_carlo_returns(policy=policy, distr=self.target_distr, n=self.n_contexts_for_eval)

            contexts = self.current_distr.normalize_samples(contexts)  # normalize samples linearly from [low, high] to [0, 1]
            contexts_ = torch.tensor(contexts).detach()
            if self.verbose >= 2:
                print(f"MC policy returns time (s): {round(time.time() - start, 2)}")


            # Handle used timesteps
            if self.count_eval_ts_in_budget:
                self.budget -= np.sum(timesteps)
            self.cum_timesteps_for_eval += np.sum(timesteps)
            wandb.log({"cum_timesteps_for_eval": self.cum_timesteps_for_eval, "timestep": self.current_timestep})
            if self.verbose >= 2:
                print('Cum timesteps used for eval:', self.cum_timesteps_for_eval, f'| {self.budget} remaining')


            # Update cumulative reward statistics
            R = torch.FloatTensor(returns).flatten()

            if self.baseline is not None:
                R -= self.baseline

            self.R_mean = self.beta * R.mean(0) + (1 - self.beta) * self.R_mean
            self.R_std = self.beta * R.std(0) + (1 - self.beta) * self.R_std

            if self.whiten_performance:
                # Whitened objective
                R_ = (R - self.R_mean) / (self.R_std + 1e-8)
            elif self.standardized_performance:
                # Use current batch of returns for standardization
                R_ = (R - R.mean(0)) / R.std(0)
            else:
                R_ = R

            if self.verbose >= 2:
                print(f'MC policy returns: {np.mean(returns)} +/- {np.std(returns)}')
                print(f'"Moving average" standardized MC policy returns: {torch.mean((R - self.R_mean) / (self.R_std + 1e-8))} +/- {torch.std((R - self.R_mean) / (self.R_std + 1e-8))} | running stats: {self.R_mean} +/- {self.R_std}')
                print(f'"In-batch" standardized MC policy returns: {torch.mean((R - R.mean(0)) / R.std(0))} +/- {torch.std((R - R.mean(0)) / R.std(0))} | in-batch stats: {R.mean(0)} +/- {R.std(0)}')

            # Update p_phi(z)
            start = time.time()
            for k in range(self.distr_learning_iters):
                # print(f'Iter {k}/{self.distr_learning_iters} | Curr distr params:', self.current_distr.get_to_params())

                # zero gradients
                self.distr_optimizer.zero_grad()

                # compute log probs and regularizer
                log_prob = self.current_distr.pdf(contexts_,
                                                  log=True,
                                                  requires_grad=True)

                if torch.any(torch.isnan(log_prob)):
                    print('Contexts ', contexts)
                    raise ValueError('Got Nan log probs')


                if self.use_kl_regularizer:
                    # Empirical kl divergence computation
                    z = self.target_distr.sample(1000000).detach()
                    log_p_train = self.current_distr.pdf(z, log=True, requires_grad=True)
                    log_p_test = self.target_distr.pdf(z, log=True, requires_grad=True)
                    kl_samples = log_p_test - log_p_train
                    kl_loss = kl_samples.mean(0)
                    if kl_loss.dim() > 0:
                        # same as before: assuming independence (why though?)
                        kl_loss = kl_loss.sum(-1)

                    regularizer = -kl_loss
                else:
                    # cross-entropy term does not influence the opt. problem
                    entropy = self.current_distr.entropy()
                    regularizer = entropy

                loss = -((R_.detach() * log_prob).mean(0) +
                         self.alpha*regularizer)
                loss.backward()
                self.distr_optimizer.step()

                # Needed to update the MultivariateNormal parameters after
                # the opt. parameters have changed.
                # Also needed in order to update the distribution for the training env
                self.current_distr.update_distribution()

            curr_entropy = self.current_distr.entropy().item()
            wandb.log({"entropy": curr_entropy, 'timestep': self.current_timestep})
            if self.verbose >= 1:
                print('Entropy after update:', curr_entropy)

            wandb.log({"loss_obj_fun": loss.item(), 'timestep': self.current_timestep})
            if self.verbose >= 2:
                print(f"Obj fun gradient descent time (s): {round(time.time() - start, 2)}")
                print('Loss obj fun:', loss.item())

            self.current_iter += 1

        self.final_policy = deepcopy(policy)

        if self.verbose >= 2:
            print('-'*50)
            print(f'Total number of training timesteps: {self.cum_timesteps_for_training}')
            print(f'Total number of MC eval timesteps: {self.cum_timesteps_for_eval}')

        return

    def is_there_budget_for_iter(self):
        return self.budget >= self.min_training_steps

    def get_monte_carlo_returns(self, policy, distr, n):
        """Get MonteCarlo returns with given policy.
            Sample contexts according to distr
        """
        self.training_subrtn.env.env_method('set_expose_episode_stats', **{'flag': True})
        self.training_subrtn.env.env_method('reset_buffer')
        target_mean_reward, target_std_reward = self.training_subrtn.eval(policy, self.target_distr, self.training_subrtn.env, eval_episodes=n)
        
        contexts, returns, timesteps = self._format_dynamics_return_timesteps_buffer(self.training_subrtn.env.env_method('get_buffer')) # returns = np.array(self._flatten(self.training_subrtn.env.env_method('get_buffer')))  # retrieve tracked returns
        succ_metrics = np.array(self._flatten(self.training_subrtn.env.env_method('get_succ_metric_buffer')))  # retrieve custom metric for measuring success, and flatten values

        if len(succ_metrics) > 0 and not self.force_success_with_returns:
            # Use returns for measuring success
            returns = succ_metrics

        self.training_subrtn.env.env_method('set_expose_episode_stats', **{'flag': False})
        self.training_subrtn.env.env_method('reset_buffer')

        assert contexts.shape[0] == returns.shape[0]

        # Depending on the args.now param, the eval episodes may differ due to sb3 implementation
        contexts = contexts[:n, :]
        returns = returns[:n]
        timesteps = timesteps[:n]

        return contexts, returns, timesteps


    def _format_dynamics_return_timesteps_buffer(self, buffer):
        """Format episode statistics (dynamics, return, timesteps)
           associated with each eval episode
        """
        dynamics = []
        returns = []
        timesteps = []

        for buffer in buffer:
            for episode_stats in buffer:
                dynamics.append(episode_stats['dynamics'])
                returns.append(episode_stats['return'])
                timesteps.append(episode_stats['timesteps'])

        return np.array(dynamics, dtype=np.float64), np.array(returns), np.array(timesteps)


    def test_on_target_distr(self, policy, ckpt_dir=None):
        """Evaluate policy on target distribution and
            keep track of best overall policy found

            ckpt_dir: save best in this dir if provided
        """
        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': True})
        self.training_subrtn.eval_env.env_method('reset_buffer')
        target_mean_reward, target_std_reward = self.training_subrtn.eval(policy, self.target_distr, self.training_subrtn.eval_env, eval_episodes=self.test_episodes)
        
        returns = np.array(self._flatten(self.training_subrtn.eval_env.env_method('get_buffer')))  # retrieve tracked returns
        succ_metrics = np.array(self._flatten(self.training_subrtn.eval_env.env_method('get_succ_metric_buffer')))  # retrieve custom metric for measuring success, and flatten values

        if len(succ_metrics) == 0 or self.force_success_with_returns:
            # Use returns for measuring success
            target_success_rate = torch.tensor(returns >= self.performance_lower_bound, dtype=torch.float32).mean()
        else:
            # Use custom metric for measuring success
            target_success_rate = torch.tensor(succ_metrics >= self.performance_lower_bound, dtype=torch.float32).mean()

        self.training_subrtn.eval_env.env_method('set_expose_episode_stats', **{'flag': False})
        self.training_subrtn.eval_env.env_method('reset_buffer')

        if target_success_rate > self.best_policy_succ_rate:
            self.best_policy_return = target_mean_reward
            self.best_policy_succ_rate = target_success_rate
            # self.best_policy = deepcopy(policy)
            self.best_distr = MultivariateGaussian(self.current_distr.get(), now=self.current_distr.now)
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

    def _save_checkpoint(self, save_dir, additional_keys: Dict = {}):
        """Save LSDR checkpoint"""
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
            return the current number of env timesteps consumed
        """
        return self.init_budget - self.budget