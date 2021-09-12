import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from npsem.NIPS2018POMIS_exp.test_bandit_strategies import load_result, compute_cumulative_regret, compute_optimality
from npsem.utils import with_default
from npsem.viz_util import sparse_index

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{helvet}\usepackage{sansmath}\sansmath"

c__ = sns.color_palette('Set1', 4)
COLORS = [c__[0], c__[0], c__[1], c__[1], c__[2], c__[2], c__[3], c__[3]]


def naked_MAB_regret_plot(axes: Axes, xs_dict, cut_time, band_alpha=0.1, legend=False, hide_ylabel=False, adjust_ymax=1, hide_yticklabels=False, **_kwargs):
    for i, (name, value_matrix) in list(enumerate(xs_dict.items())):
        mean_x = np.mean(value_matrix, axis=0)
        sd_x = np.std(value_matrix, axis=0)
        lower, upper = mean_x - sd_x, mean_x + sd_x

        time_points = sparse_index(with_default(cut_time, len(mean_x)), 200)
        axes.plot(time_points, mean_x[time_points], lw=1, label=name.split(' ')[0] if '(TS)' in name else None, color=COLORS[i], linestyle='-' if '(TS)' in name else '--')
        axes.fill_between(time_points, lower[time_points], upper[time_points], color=COLORS[i], alpha=band_alpha, lw=0)

    if legend:
        axes.legend(loc=2, frameon=False)
    if not hide_ylabel:
        axes.set_ylabel('Cum. Regrets')
        axes.get_yaxis().set_label_coords(-0.15, 0.5)
    if adjust_ymax != 1:
        ymin, ymax = axes.get_ylim()
        axes.set_ylim(ymin, ymax * adjust_ymax)
    if hide_yticklabels:
        axes.set_yticklabels([])


def naked_MAB_optimal_probability_plot(axes: Axes, arm_freqs, cut_time, legend=False, hide_ylabel=False, hide_yticklabels=False, **_kwargs):
    for i, (name, arm_freq) in list(enumerate(arm_freqs.items())):
        time_points = sparse_index(with_default(cut_time, len(arm_freq)), 200)
        axes.plot(time_points, arm_freq[time_points], lw=1, label=name.split(' ')[0] if '(TS)' in name else None, color=COLORS[i], linestyle='-' if '(TS)' in name else '--')
    if legend:
        axes.legend(loc=4, frameon=False)
    axes.set_xlabel('Trials')
    if not hide_ylabel:
        axes.set_ylabel('Probability')
        axes.get_yaxis().set_label_coords(-0.15, 0.5)
    axes.set_yticks([0, 0.5, 1.0])
    if hide_yticklabels:
        axes.set_yticklabels([])

    axes.set_ylim(-0.05, 1.02)


def data_prep(directory):
    _, mu, results = load_result(directory)
    mu_star = np.max(mu)

    regret_results = dict()
    arm_optimality_results = dict()

    # prepare data
    for (arm_strategy, bandit_algo), (arm_played, rewards) in results.items():
        legend_label = arm_strategy + ' (' + bandit_algo + ')'

        cumulative_regret = compute_cumulative_regret(rewards, mu_star)
        arm_optimality = compute_optimality(arm_played, mu)

        regret_results[legend_label] = cumulative_regret
        arm_optimality_results[legend_label] = np.mean(arm_optimality, axis=0)

    return regret_results, arm_optimality_results


def aggregate_plot():
    """ Prepare data """
    info__ = {"Markovian": (absolute_path + f'bandit_results/mark_0', 1000), "IV": (absolute_path + f'bandit_results/iv_0', 1000), "XYZWST": (absolute_path + f'bandit_results/xyzwst_0', 10000)}
    info = {k: dict(zip(['directory', 'cut_time'], v)) for k, v in info__.items()}
    results = {task_name: dict(zip(['CR', 'OAP'], data_prep(task_info['directory']))) for task_name, task_info in info.items()}
    plot_funcs = {'CR': naked_MAB_regret_plot, 'OAP': naked_MAB_optimal_probability_plot}

    """ Start drawing """
    fig, ax = plt.subplots(2, 3, sharex='col', figsize=(8, 3.25))
    for row_id, plot_type in enumerate(['CR', 'OAP']):
        for col_id, task_name in enumerate(['Markovian', 'IV', 'XYZWST']):
            current_axes = ax[row_id, col_id]
            plot_funcs[plot_type](current_axes, results[task_name][plot_type], info[task_name]['cut_time'],
                                  legend=row_id == 1 and col_id == 0,
                                  hide_ylabel=col_id != 0,
                                  hide_yticklabels=False,
                                  adjust_ymax={'Markovian': 1, 'IV': 0.35, 'XYZWST': 0.55}[task_name] if plot_type == 'CR' else 1)

    sns.despine(fig)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.175)
    fig.savefig('aggregate.pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    absolute_path = ''
    aggregate_plot()
