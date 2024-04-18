from . import core
from . import gym_envs

# Set the ordering convention for Euler angles
# EULER_ORDER = "zyx"  # old
EULER_ORDER = "xyz"  # default Mujoco convention (eulerseq at https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler)

