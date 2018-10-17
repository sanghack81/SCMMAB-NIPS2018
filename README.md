# Structural Causal Bandits: Where to Intervene?

Sanghack Lee and Elias Bareinboim Structural Causal Bandits: Where to Intervene? In _Advances in Neural Information Processing System 31_, _forthcoming_  2018

We provide codebase to allow readers to reproduce our experiments. This code also contains various utilities related to causal diagram, structural causal model, and multi-armed bandit problem.
(At this moment, the code is not well-documented.) 
The code is tested with the following configuration: `python=3.7`, `numpy=1.15.2`, `scipy=1.1.0`, `joblib=0.12.5`, `matplotlib=3.0.0`, `seaborn=0.9.0`, and `networkx=2.2`, on
Linux and MacOS machines.


Please run the following command to perform experiments (which will use 3/4 of CPU cores on the machine):
> `python -m npsem.NIPS2018POMIS_exp.test_bandit_strategies`

This takes less than 30 minutes in a server with 24 cores. This will create `bandit_results` directory and there will be three directories corresponding to each task in the paper.
Then, run the following to create a figure as in the paper:
> `python -m npsem.NIPS2018POMIS_exp.test_drawing_re`




