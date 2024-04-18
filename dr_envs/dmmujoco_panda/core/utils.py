import pdb

from gym.envs import register
import numpy as np

def soft_tanh_limit(x, low, high, betas=(0.35, 0.35), linear_coeff=1e-3, square_coeff=0.3):
    norm_x = (x - low) / (high - low)
    alpha_min, alpha_max = norm_x, 1-norm_x
    penalty = np.zeros_like(x)
    val_low = .5 * (1 - np.tanh(1/(1-alpha_min/betas[0]) - betas[0]/alpha_min))
    val_high = .5 * (1 - np.tanh(1/(1-alpha_max/betas[1]) - betas[1]/alpha_max))
    idx_low = alpha_min < betas[0]
    idx_high = alpha_max < betas[1]
    penalty[idx_low] = val_low[idx_low]
    penalty[idx_high] = val_high[idx_high]
    penalty[alpha_min < 0] = 1
    penalty[alpha_max < 0] = 1

    linear_term = (np.fmax(0, alpha_min) + np.fmax(0, alpha_max))
    square_term = np.fmin((norm_x*2-1)**2, 1.)

    return square_term*square_coeff + penalty*(1-square_coeff) + linear_coeff * linear_term

def square_penalty_limit(x, low, high, square_coeff=0.5):
    norm_x = (x - low) / (high - low)
    square_term = np.fmin(square_coeff*(norm_x*2-1)**2, 1.)
    return square_term

def distance_penalty(d, w=1, v=1, alpha=1e-3):
    return -w*d**2 - v*np.log(d**2 + alpha)


# def get_dim(param):
#     """
#     :description: Returns the dimension of ``param``
#     :param param: the array/float value to get dimensionality of
#     :return: the number of elements in ``param``
#     """
#     if isinstance(param, (float, int)):
#         return 1
#     elif isinstance(param, np.ndarray):
#         return len(param)
#     elif param == None:
#         return 0
#     else:
#         raise TypeError(f"Don't know how to handle {param} of type {type(param)}")


def register_panda_env(id,
                       entry_point,
                       model_file: str,
                       model_kwargs: dict,
                       action_interpolator,
                       action_interpolator_kwargs: dict,
                       controller,
                       controller_kwargs: dict,
                       env_kwargs: dict,
                       **kwargs):
    """
    :description: Register a new Panda environment, such that it can be instantiated
        with a call to ``gym.make``.
    :param id: The name used for registration,
    :param entry_point: The Python class of the environment. It has to include
        the module name (e.g. ``reach:PandaReachEnv``; when calling from the
        same file where the class is declared, it's best to simply use
        ``"%s:ClassName" % __name__``)
    """
    register(id=id,
             entry_point=entry_point,
             kwargs={"model_file": model_file,
                     "model_kwargs": model_kwargs,
                     "action_interpolator": action_interpolator,
                     "action_interpolator_kwargs": action_interpolator_kwargs,
                     "controller": controller,
                     "controller_kwargs": controller_kwargs,
                     **env_kwargs},
             **kwargs)
