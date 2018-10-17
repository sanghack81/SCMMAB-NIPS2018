import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

from npsem.NIPS2018POMIS_exp.test_bandit_strategies import load_result, compute_cumulative_regret, compute_optimality, compute_arm_frequencies
from npsem.utils import with_default, mkdirs
from npsem.viz_util import sparse_index

rc('text', usetex=True)
pl.rcParams['text.latex.preamble'] = [
    r'\usepackage{tgheros}',  # helvetica font
    r'\usepackage{sansmath}',  # math-font matching  helvetica
    r'\sansmath'  # actually tell tex to use it!
    r'\usepackage{siunitx}',  # micro symbols
    r'\sisetup{detect-all}',  # force siunitx to use the fonts
]


def MAB_optimal_probability_plot2(fname, arm_freqs, colors=None, title=None, fig_kwargs=None, cut_time=None, name_filter=None, name_change=None, no_legend=False):
    """ Draw figures showing optimal arm selection probability (based on multiple runs) """
    if colors is None:
        colors = sns.color_palette('Set1', len(arm_freqs))
    sns.set(style="white", font_scale=1.4, rc={"lines.linewidth": 20})
    plt.figure(**with_default(fig_kwargs, dict()))
    for i, (name, arm_freq) in enumerate(arm_freqs.items()):
        if name_filter is not None:
            if not name_filter(name):
                continue
        time_points = sparse_index(with_default(cut_time, len(arm_freq)))
        plt.plot(time_points, arm_freq[time_points], lw=1, label=name_change[name], color=colors[i], linestyle='-' if 'once' in name else "--")
    if not no_legend:
        plt.legend(loc=4)
    plt.xlabel('Trials')
    plt.ylabel('Opt. Arm Prob.')
    plt.xticks([])
    plt.ylim(-0.05, 1.02)
    if title is not None:
        plt.title(title)
    for l in plt.gca().lines:
        print(l.get_linewidth())
        plt.setp(l, linewidth=2)
    for line in plt.gca().legend().get_lines():
        line.set_linewidth(2)
    sns.despine()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def MAB_regret_plot2(fname, xs_dict, colors=None, title=None, fig_kwargs=None, cut_time=None, name_filter=None, name_change=None):
    if colors is None:
        colors = sns.color_palette('Set1', len(xs_dict))
    sns.set(style="white", font_scale=1.4, rc={"lines.linewidth": 20})
    plt.figure(**with_default(fig_kwargs, dict()))

    for i, (name, value_matrix) in enumerate(xs_dict.items()):
        if name_filter is not None:
            if not name_filter(name):
                continue
        mean_x = np.mean(value_matrix, axis=0)

        time_points = sparse_index(with_default(cut_time, len(mean_x)))
        plt.plot(time_points, mean_x[time_points], lw=1, label=name_change[name], color=colors[i], linestyle='-' if 'once' in name else "--")

    plt.xlabel('Trials')
    plt.ylabel('Cum. Regrets')
    plt.xticks([])
    plt.yticks([])
    plt.ylim([-5, 100])
    if title is not None:
        plt.title(title)
    for l in plt.gca().lines:
        print(l.get_linewidth())
        plt.setp(l, linewidth=2)

    sns.despine()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.02)
    plt.close()


def main_viz2(directory, results, mu, horizons):
    """ Visualization method """
    mkdirs(directory)
    mu_star = np.max(mu)

    regret_results = dict()
    arm_optimality_results = dict()
    arm_counts_results = dict()

    # prepare data
    for (arm_strategy, bandit_algo), (arm_played, rewards) in results.items():
        legend_label = arm_strategy + ' (' + bandit_algo + ')'

        cumulative_regret = compute_cumulative_regret(rewards, mu_star)
        arm_optimality = compute_optimality(arm_played, mu)
        arm_counts = compute_arm_frequencies(arm_played, len(mu))

        regret_results[legend_label] = cumulative_regret
        arm_optimality_results[legend_label] = np.mean(arm_optimality, axis=0)
        arm_counts_results[legend_label] = arm_counts

    # visualize data w/ different horizon settings
    colors = sns.color_palette('Set1', 4)
    colors = [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2], colors[3], colors[3]]
    for cut in horizons:
        MAB_optimal_probability_plot2(directory + f'/toy_prob_{str(cut).zfill(5)}.pdf', arm_optimality_results, colors, cut_time=cut, fig_kwargs={'figsize': (3.5, 2)},
                                      name_filter=lambda x: "(TS)" in x and "MIS" not in x,
                                      name_change={'Brute-force (TS)': 'All Subsets', 'All-at-once (TS)': 'All-at-once'})
        MAB_regret_plot2(directory + f'/toy_regret_{str(cut).zfill(5)}.pdf', regret_results, colors, fig_kwargs={'figsize': (3.5, 2)}, cut_time=cut,
                         name_filter=lambda x: "(TS)" in x and "MIS" not in x, name_change={'Brute-force (TS)': 'All Subsets', 'All-at-once (TS)': 'All-at-once'})


def main():
    directory = f'bandit_results/iv_0'
    _p_u, _mu, _results = load_result(directory)
    fig_directory = f'toy_figs'
    main_viz2(fig_directory, _results, _mu, [1000])


if __name__ == '__main__':
    # please run after running test_bandit_strategies.py
    main()
