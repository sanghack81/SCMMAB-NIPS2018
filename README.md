# Structural Causal Bandits: Where to Intervene?

*Sanghack Lee and Elias Bareinboim* Structural Causal Bandits: Where to Intervene? In _Advances in Neural Information Processing System 31 (NIPS'2018), 2018

We provide codebase to allow readers to reproduce our experiments. This code also contains various utilities related to causal diagram, structural causal model, and multi-armed bandit problem.
(At this moment, the code is not well-documented.) 
The codebase is actively modernized to be compatible with **Python >= 3.11** (the minimum required by the current `numpy`/`scipy`/`networkx` stack) up to the latest environments (tested on Python 3.14).
It relies on standard scientific libraries, and you don't need strictly pinned versions. It is tested seamlessly with newer configurations of `numpy`, `scipy`, `joblib`, `matplotlib`, `seaborn`, and `networkx` on Linux and MacOS.


Please run the following command to perform experiments (which will use 3/4 of CPU cores on the machine):
> `python3 -m npsem.NIPS2018POMIS_exp.test_bandit_strategies`

This takes less than 30 minutes in a server with 24 cores. This will create `bandit_results` directory and there will be three directories corresponding to each task in the paper.
Then, run the following to create a figure as in the paper:
> `python3 -m npsem.NIPS2018POMIS_exp.test_drawing_re`

## Running the tests

Install the test dependency and run the unit-test suite:
> `pip install -e .[test]`
>
> `pytest`

The suite cross-checks the POMIS algorithm against a brute-force oracle, verifies the
paper's optimality claim, and guards the bandit algorithms; it runs in a few seconds.
