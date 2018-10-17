import multiprocessing
import numpy as np
import os
from pathlib import Path

from npsem.NIPS2018POMIS_exp.scm_examples import XYZWST_SCM, simple_markovian_SCM, IV_SCM
from npsem.bandits import play_bandits
from npsem.model import StructuralCausalModel
from npsem.scm_bandits import SCM_to_bandit_machine, arms_of, arm_types
from npsem.utils import subseq, mkdirs


def main_experiment(M: StructuralCausalModel, Y='Y', num_trial=200, horizon=10000, n_jobs=1):
    # run bandit algorithms
    results = dict()
    mu, arm_setting = SCM_to_bandit_machine(M)
    for arm_strategy in arm_types():
        arm_selected = arms_of(arm_strategy, arm_setting, M.G, Y)
        arm_corrector = np.vectorize(lambda x: arm_selected[x])
        for bandit_algo in ['TS', 'UCB']:
            arm_played, rewards = play_bandits(horizon, subseq(mu, arm_selected), bandit_algo, num_trial, n_jobs)
            results[(arm_strategy, bandit_algo)] = arm_corrector(arm_played), rewards

    return results, mu


def compute_arm_frequencies(arm_played, num_arms, horizon=None):
    if horizon is not None:
        arm_played = arm_played[:, :horizon]

    counts = np.zeros((len(arm_played), num_arms))
    for i in range(num_arms):
        counts[:, i] = np.mean((arm_played == i).astype(int), axis=1)
    return counts


def compute_optimality(arm_played, mu):
    mu_star = np.max(mu)
    return np.vectorize(lambda x: int(mu[x] == mu_star))(arm_played)


def compute_cumulative_regret(rewards: np.ndarray, mu_star: float) -> np.ndarray:
    cumulative_rewards = np.cumsum(rewards, axis=1)
    optimal_cumulative_rewards = np.cumsum(np.ones(rewards.shape) * mu_star, axis=1)
    cumulative_regret = optimal_cumulative_rewards - cumulative_rewards
    return cumulative_regret


def load_result(directory):
    results = dict()
    for arm_strategy in arm_types():
        for bandit_algo in ['TS', 'UCB']:
            loaded = np.load(directory + f'/{arm_strategy}---{bandit_algo}.npz')
            arms = loaded['a']
            rewards = loaded['b']
            results[(arm_strategy, bandit_algo)] = (arms, rewards)

    p_u = np.load(directory + '/p_u.npz')['a'][()]
    mu = tuple(np.load(directory + '/mu.npz')['a'])
    return p_u, mu, results


def save_result(directory, p_u, mu, results):
    mkdirs(directory)
    for arm_strategy, bandit_algo in results:
        arms, rewards = results[(arm_strategy, bandit_algo)]
        np.savez_compressed(directory + f'/{arm_strategy}---{bandit_algo}', a=arms, b=rewards)
    np.savez_compressed(directory + f'/p_u', a=p_u)
    np.savez_compressed(directory + f'/mu', a=mu)


def finished(directory, flag=None, message=''):
    mkdirs(directory)
    filename = directory + '/finished.txt'
    if flag is not None:
        if flag:
            Path(filename).touch(exist_ok=True)
            with open(filename, 'w') as f:
                print(str(message), file=f)
            return True
        else:
            os.remove(filename)
            return False
    else:
        return os.path.exists(filename)


def main():
    """ Conduct experiments and draw figures. If experiments are already done, draw graphs again."""
    num_simulation_repeats = 300
    for dirname, (model, p_u), horizon in [('xyzwst', XYZWST_SCM(True, seed=0), 10000),
                                           ('mark', simple_markovian_SCM(seed=0), 10000),
                                           ('iv', IV_SCM(True, seed=0), 5000)]:
        directory = f'bandit_results/{dirname}_0'
        if not finished(directory):
            results, mu = main_experiment(model, 'Y', num_simulation_repeats, horizon, n_jobs=3 * multiprocessing.cpu_count() // 4)
            save_result(directory, p_u, mu, results)
            finished(directory, flag=True)


if __name__ == '__main__':
    main()
