# DORAEMON: Domain Randomization via Entropy Maximization

[Paper](https://arxiv.org/abs/2311.01885) / [Website](https://gabrieletiboni.github.io/doraemon/) / [Video](https://gabrieletiboni.github.io/doraemon/)

##### Gabriele Tiboni, Pascal Klink, Jan Peters, Tatiana Tommasi, Carlo D'Eramo, Georgia Chalvatzaki

This repository contains the code for the paper "Domain Randomization via Entropy Maximization".

*Abstract:* Varying dynamics parameters in simulation is a popular Domain Randomization (DR) approach for overcoming the reality gap in Reinforcement Learning (RL). Nevertheless, DR heavily hinges on the choice of the sampling distribution of the dynamics parameters, since high variability is crucial to regularize the agent's behavior but notoriously leads to overly conservative policies when randomizing excessively.
In this paper, we propose a novel approach to address sim-to-real transfer, which automatically shapes dynamics distributions during training in simulation without requiring real-world data.
We introduce DOmain RAndomization via Entropy MaximizatiON (DORAEMON), a constrained optimization problem that directly maximizes the entropy of the training distribution while retaining generalization capabilities. In achieving this, DORAEMON gradually increases the diversity of sampled dynamics parameters as long as the probability of success of the current policy is sufficiently high.
We empirically validate the consistent benefits of DORAEMON in obtaining highly adaptive and generalizable policies, i.e. solving the task at hand across the widest range of dynamics parameters, as opposed to representative baselines from the DR literature. Notably, we also demonstrate the Sim2Real applicability of DORAEMON through its successful zero-shot transfer in a robotic manipulation setup under unknown real-world parameters.

![Alt Text](docs/assets/img/gifs/AutoDR_medium_progress.gif)

## Installation

1. Install mujoco 2.1 (follow instructions below or [here](https://github.com/openai/mujoco-py)):
```
# 1.1
# Install mujoco binaries
cd ~
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz 
mkdir ~/.mujoco
mv ~/mujoco210-linux-x86_64.tar.gz ~/.mujoco
cd ~/.mujoco
tar -xf mujoco210-linux-x86_64.tar.gz

# 1.2
# Install mujoco 2.1 dependencies at:
# https://github.com/openai/mujoco-py/issues/627

# 1.3
# Add env variables (path may differ for your system)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

2. Install DORAEMON dependences:
```
pip install -r requirements.txt
```

3. Install the [exps-launcher](https://github.com/gabrieletiboni/exps-launcher/) package. You may skip this step and use classic CLI parameters instead, but we provide examples based on our experiments-launcher syntax, as it improves readability.

4. (optional) set up `wandb login` with your WeightsAndBiases account. If you do not wish to use wandb to track the experiment results, set `wandb="disabled"` when launching our scripts with the exps-launcher.  

This code has been tested with Python 3.7, on a Ubuntu 20.04 machine.

## Getting started

### Quick test
- `python train_doraemon.py --env RandomContinuousInvertedCartPoleEasy-v0 -t 1500 --eval_freq 500 --gradient_steps 1 --eval_episodes 1 --test_episodes 1 --seed 42 --dr_percentage 0.5 --algo sac --performance_lb 0 --kl_ub 2 --n_iters 3 --verbose 2 --wandb disabled --debug`

### Reproduce paper results
The commands below launch training scripts locally in background, on a single seed. Alternatively, you may set the flag `exps.noslurm=false` to run them as slurm jobs. In the latter case, you may specify flags such as `host.time=300` (5 hours) for custom sbatch parameters.
To run multiple seeds, you may add the flag `sweep.config=[tenseeds]` or `sweep.seed=[42,43,44,45,46]`. Refer to [exps-launcher](https://github.com/gabrieletiboni/exps-launcher/) for more information on our experiments-launcher syntax.

#### DORAEMON
- Inclined Plane (toy problem): `python launch_exps.py script=train_doraemon config=[adaptive+,plane_v2,short,succRate50] sweep.kl_ub=[0.1,0.05,0.01,0.005,0.001] seed=42 wandb="online" exps.noslurm=true`
- CartPole: `python launch_exps.py script=train_doraemon config=[adaptive+,classicCartpoleEasy,short,succRate50] seed=42 exps.noslurm=true wandb="online" kl_ub=0.001`
- SwingUpCartPole: `python launch_exps.py script=train_doraemon config=[adaptive+,cartpoleEasy,short,succRate50] seed=42 exps.noslurm=true wandb="online" kl_ub=0.1`
- Hopper: `python launch_exps.py script=train_doraemon config=[adaptive+,hopper,longx,succRate50] seed=42 exps.noslurm=true wandb="online" kl_ub=0.005`
- Walker2d: `python launch_exps.py script=train_doraemon config=[adaptive+,walker2d_v2,longx,succRate50] seed=42 exps.noslurm=true wandb="online" kl_ub=0.01`
- HalfCheetah: `python launch_exps.py script=train_doraemon config=[adaptive+,halfcheetah,longx,succRate50] seed=42 exps.noslurm=true wandb="online" kl_ub=0.005`
- Swimmer: `python launch_exps.py script=train_doraemon config=[adaptive+,swimmer_v2,longx,succRate50] seed=42 exps.noslurm=true wandb="online" kl_ub=0.001`
- PandaPush: `python launch_exps.py script=train_doraemon config=[adaptive+,pandaPush_v11,medium,succRate50,easyStart,qacc04] seed=42 exps.noslurm=true wandb="online" kl_ub=0.1`

Check out `reproduce_paper_results.md` file for the complete command list to reproduce all baselines and experiments in the paper.


## DORAEMON on your Custom Gym Environments

DORAEMON builds on top of gym.Env environments (gym==0.21.0), but further relies on environment-specific methods for handling the Domain Randomization distribution. This codebase follows the convention introduced in the [random-envs](https://github.com/gabrieletiboni/random-envs) package (see `dev` branch for compatibility with this codebase), hence can be used with any of the environments available there. If you wish to launch DORAEMON on your own custom gym environment, you must implement the missing custom methods (e.g. `get_task()`, `set_task()`, ...). Check out the [random-envs](https://github.com/gabrieletiboni/random-envs) codebase to view the expected interface.


## Cite us
If you use this repository, please consider citing
```
@misc{tiboni2023doraemon,
  title={Domain Randomization via Entropy Maximization}, 
  author={Gabriele Tiboni and Pascal Klink and Jan Peters and Tatiana Tommasi and Carlo D'Eramo and Georgia Chalvatzaki},
  year={2023},
  eprint={2311.01885},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```