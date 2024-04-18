# Reproduce paper results

The commands below launch training scripts locally in background, on a single seed. Alternatively, you may set the flag `exps.noslurm=false` to run them as slurm jobs. In the latter case, you may specify flags such as `host.time=300` (5 hours) for custom sbatch parameters.
To run multiple seeds, you may add the flag `sweep.config=[tenseeds]` or `sweep.seed=[42,43,44,45,46]`.
Refer to [exps-launcher](https://github.com/gabrieletiboni/exps-launcher/) for more information on our experiments-launcher syntax.

#### NO-DR
  - CartPole: `python launch_exps.py script=train_udr config=[adaptive+,classicCartpoleEasy,short] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_NoDR" dr_percentage=0.0`
  - SwingUpCartPole: `python launch_exps.py script=train_udr config=[adaptive+,cartpoleEasy,short] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_NoDR" dr_percentage=0.0`
  - Hopper: `python launch_exps.py script=train_udr config=[adaptive+,hopper,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_NoDR" dr_percentage=0.0`
  - Walker2d: `python launch_exps.py script=train_udr config=[adaptive+,walker2d_v2,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_NoDR" dr_percentage=0.0`
  - HalfCheetah: `python launch_exps.py script=train_udr config=[adaptive+,halfcheetah,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_NoDR" dr_percentage=0.0`
  - Swimmer: `python launch_exps.py script=train_udr config=[adaptive+,swimmer_v2,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_NoDR" dr_percentage=0.0`
  - PandaPush: `python launch_exps.py script=train_udr config=[adaptive+,pandaPush_v11,medium,easyDR0,qacc04] seed=42 exps.noslurm=true wandb="offline"`

#### Fixed-DR
 - CartPole: `python launch_exps.py script=train_udr config=[adaptive+,classicCartpoleEasy,short] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_FixedDR"`
 - SwingUpCartPole: `python launch_exps.py script=train_udr config=[adaptive+,cartpoleEasy,short] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_FixedDR"`
 - Hopper: `python launch_exps.py script=train_udr config=[adaptive+,hopper,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_FixedDR"`
 - Walker2d: `python launch_exps.py script=train_udr config=[adaptive+,walker2d_v2,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_FixedDR"`
 - HalfCheetah: `python launch_exps.py script=train_udr config=[adaptive+,halfcheetah,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_FixedDR"`
 - Swimmer: `python launch_exps.py script=train_udr config=[adaptive+,swimmer_v2,longx] seed=42 wandb=offline exps.noslurm=true exps.group_suffix="_FixedDR"`
 - PandaPush: `python launch_exps.py script=train_udr config=[adaptive+,pandaPush_v11,medium,space1,qacc04] seed=42 exps.noslurm=true wandb="offline"`

#### DORAEMON
- Inclined Plane (toy problem): `python launch_exps.py script=train_doraemon config=[adaptive+,plane_v2,short,succRate50] sweep.kl_ub=[0.1,0.05,0.01,0.005,0.001] seed=42 wandb="offline" exps.noslurm=true`
- CartPole: `python launch_exps.py script=train_doraemon config=[adaptive+,classicCartpoleEasy,short,succRate50] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.001`
- SwingUpCartPole: `python launch_exps.py script=train_doraemon config=[adaptive+,cartpoleEasy,short,succRate50] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.1`
- Hopper: `python launch_exps.py script=train_doraemon config=[adaptive+,hopper,longx,succRate50] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.005`
- Walker2d: `python launch_exps.py script=train_doraemon config=[adaptive+,walker2d_v2,longx,succRate50] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.01`
- HalfCheetah: `python launch_exps.py script=train_doraemon config=[adaptive+,halfcheetah,longx,succRate50] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.005`
- Swimmer: `python launch_exps.py script=train_doraemon config=[adaptive+,swimmer_v2,longx,succRate50] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.001`
- PandaPush: `python launch_exps.py script=train_doraemon config=[adaptive+,pandaPush_v11,medium,succRate50,easyStart,qacc04] seed=42 exps.noslurm=true wandb="offline" kl_ub=0.1`

#### AutoDR
- CartPole: `python launch_exps.py script=train_autodr config=[adaptive+,classicCartpoleEasy,short,original] exps.noslurm=true seed=42 wandb="offline" delta=0.0166`
- SwingUpCartPole: `python launch_exps.py script=train_autodr config=[adaptive+,cartpoleEasy,short,original] exps.noslurm=true seed=42 wandb="offline" delta=0.0498`
- Hopper: `python launch_exps.py script=train_autodr config=[adaptive+,hopper,longx,original] exps.noslurm=true seed=42 wandb="offline" delta=0.0083`
- Walker2d: `python launch_exps.py script=train_autodr config=[adaptive+,walker2d_v2,longx,original] exps.noslurm=true seed=42 wandb="offline" delta=0.0166`
- HalfCheetah: `python launch_exps.py script=train_autodr config=[adaptive+,halfcheetah,longx,original] exps.noslurm=true seed=42 wandb="offline" delta=0.0083`
- Swimmer: `python launch_exps.py script=train_autodr config=[adaptive+,swimmer_v2,longx,original] exps.noslurm=true seed=42 wandb="offline" delta=0.0083`
- PandaPush: `python launch_exps.py script=train_autodr config=[adaptive+,pandaPush_v11,medium,original,easyStart,qacc04] seed=42 exps.noslurm=true wandb="offline" delta=0.0498`


#### LSDR
- CartPole: `python launch_exps.py script=train_lsdr config=[adaptive+,classicCartpoleEasy,short,slowPace] exps.noslurm=true seed=42 wandb=offline alpha=5`
- SwingUpCartPole: `python launch_exps.py script=train_lsdr config=[adaptive+,cartpoleEasy,short,slowPace] exps.noslurm=true seed=42 wandb=offline alpha=1`
- Hopper: `python launch_exps.py script=train_lsdr config=[adaptive+,hopper,longx,slowPace] exps.noslurm=true seed=42 wandb=offline alpha=5`
- Walker2d: `python launch_exps.py script=train_lsdr config=[adaptive+,walker2d_v2,longx,slowPace] exps.noslurm=true seed=42 wandb=offline alpha=10`
- HalfCheetah: `python launch_exps.py script=train_lsdr config=[adaptive+,halfcheetah,longx,slowPace] exps.noslurm=true seed=42 wandb=offline alpha=0.1`
- Swimmer: `python launch_exps.py script=train_lsdr config=[adaptive+,swimmer_v2,longx,slowPace] exps.noslurm=true seed=42 wandb=offline alpha=0.1`
- PandaPush: `python launch_exps.py script=train_lsdr config=[adaptive+,pandaPush_v11,medium,slowPace,easyStart,qacc04,baselinePandaReturn] seed=42 exps.noslurm=true wandb=offline alpha=0.0001 obj_fun_lr=0.001`