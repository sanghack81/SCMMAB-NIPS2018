# Structural Causal Bandits: Where to Intervene?

*Sanghack Lee and Elias Bareinboim* Structural Causal Bandits: Where to Intervene? In _Advances in Neural Information Processing System 31 (NIPS'2018), 2018

We provide codebase to allow readers to reproduce our experiments. This code also contains various utilities related to causal diagram, structural causal model, and multi-armed bandit problem.
(At this moment, the code is not well-documented.) 
The code is tested with the following configuration: `python=3.9`, `numpy=1.21.2`, `scipy=1.7.1`, `joblib=1.0.1`, `matplotlib=3.4.3`, `seaborn=0.11.2`, and `networkx=2.6.3`, on
Linux and MacOS machines.


Please run the following command to perform experiments (which will use 3/4 of CPU cores on the machine):
> `python3 -m npsem.NIPS2018POMIS_exp.test_bandit_strategies`

This takes less than 30 minutes in a server with 24 cores. This will create `bandit_results` directory and there will be three directories corresponding to each task in the paper.
Then, run the following to create a figure as in the paper:
> `python3 -m npsem.NIPS2018POMIS_exp.test_drawing_re`
