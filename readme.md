# DORAEMON: Domain Randomization via Entropy Maximization

[Preprint](https://arxiv.org/abs/2311.01885) / [Website](https://gabrieletiboni.github.io/doraemon/)

##### Gabriele Tiboni, Pascal Klink, Jan Peters, Tatiana Tommasi, Carlo D'Eramo, Georgia Chalvatzaki

This repository contains the code for the paper "Domain Randomization via Entropy Maximization".

*Abstract:* Varying dynamics parameters in simulation is a popular Domain Randomization (DR) approach for overcoming the reality gap in Reinforcement Learning (RL). Nevertheless, DR heavily hinges on the choice of the sampling distribution of the dynamics parameters, since high variability is crucial to regularize the agent's behavior but notoriously leads to overly conservative policies when randomizing excessively.
In this paper, we propose a novel approach to address sim-to-real transfer, which automatically shapes dynamics distributions during training in simulation without requiring real-world data.
We introduce DOmain RAndomization via Entropy MaximizatiON (DORAEMON), a constrained optimization problem that directly maximizes the entropy of the training distribution while retaining generalization capabilities. In achieving this, DORAEMON gradually increases the diversity of sampled dynamics parameters as long as the probability of success of the current policy is sufficiently high.
We empirically validate the consistent benefits of DORAEMON in obtaining highly adaptive and generalizable policies, i.e. solving the task at hand across the widest range of dynamics parameters, as opposed to representative baselines from the DR literature. Notably, we also demonstrate the Sim2Real applicability of DORAEMON through its successful zero-shot transfer in a robotic manipulation setup under unknown real-world parameters.

Our code release is currently **under construction**.

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