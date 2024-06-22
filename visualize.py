import os
from networkx import is_empty
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import seaborn as sns

from argparse import ArgumentParser

from constants import ANALYSIS_DIR, EXPERIMENTS_DIR, FIGSIZE, NUMBER_OF_OBJECTIVES, OPTIMISERS
from utils import count_dir, non_dominated_sorting, visualize_objective_space

if __name__ == "__main__":
    # Setup input option
    parser = ArgumentParser(description="Network-on-Chips")
    parser.add_argument('--name', type=str, required=True, help='Test case name')
    args = parser.parse_args()

    # Get input option
    testcase_name = args.name

    # Get experiments test case folder
    if not os.path.exists(ANALYSIS_DIR):
        os.mkdir(ANALYSIS_DIR)
    ROOT_DIR = os.path.join(ANALYSIS_DIR, testcase_name)
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)

    RECORDED_DIR = os.path.join(EXPERIMENTS_DIR, testcase_name)
    n_experiments = count_dir(RECORDED_DIR)
    
    final_population = {}
    nsga_ii = []
    bilevel = []
    for opt in OPTIMISERS:
        for i in range(n_experiments):
            EXPERIMENT_DIR = os.path.join(RECORDED_DIR, f'experiment_{i}')
            n_files = count_dir(EXPERIMENT_DIR, opt)
            f_exp = []
            for j in range(n_files):
                data_iter = np.loadtxt(os.path.join(EXPERIMENT_DIR, f'{opt}_fitness_{j}.txt'))
                f_exp.append(data_iter)
            if opt == 'NSGA-II':
                nsga_ii.append(f_exp)
            if opt == 'Bi-level':
                bilevel.append(f_exp)

    # Visualize the objective space
    optimal_bilevel = np.array([f_exp[-1][0] for f_exp in bilevel])
    plt.figure(figsize=(8, 8))
    plt.scatter(
        x=optimal_bilevel[:, 0],
        y=optimal_bilevel[:, 1],
        label=f'Bi-level optimal solution',
    )

    optimal_nsga_ii = np.array([])
    for f_exp in nsga_ii:
        if len(optimal_nsga_ii) == 0:
            optimal_nsga_ii = f_exp[-1]
        optimal_nsga_ii = np.append(optimal_nsga_ii, f_exp[-1], axis=0)
    pareto_fronts = non_dominated_sorting(optimal_nsga_ii)
    front = pareto_fronts[0]
    non_dominated = optimal_nsga_ii[front]

    dominated = np.array([])
    for i in range(1, len(pareto_fronts)):
        if len(dominated) == 0:
            dominated = optimal_nsga_ii[pareto_fronts[i]]
        dominated = np.append(dominated, optimal_nsga_ii[pareto_fronts[i]], axis=0)
    dominated = np.array(dominated)
    if dominated.size != 0:
        plt.scatter(
            x=dominated[:, 0],
            y=dominated[:, 1],
            label=f'NSGA-II dominated solution',
        )
    plt.scatter(
        x=non_dominated[:, 0],
        y=non_dominated[:, 1],
        label=f'NSGA-II non-dominated solution',
    )
    
    plt.title('Objective space')
    plt.xlabel('Energy Consumption')
    plt.ylabel('Average Load Degree')
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, f'objective_space'))
    plt.close()

    # Box plot
    ec_nsga_ii = np.array([])
    ec_bilevel = np.array([])
    ld_nsga_ii = np.array([])
    ld_bilevel = np.array([])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for exp in nsga_ii:
        for fitness in exp:
            ec_nsga_ii = np.append(ec_nsga_ii, fitness[:, 0])
            ld_nsga_ii = np.append(ld_nsga_ii, fitness[:, 1])

    for exp in bilevel:
        for fitness in exp:
            ec_bilevel = np.append(ec_bilevel, fitness[:, 0])
            ld_bilevel = np.append(ld_bilevel, fitness[:, 1])
    ax1.boxplot([ec_nsga_ii, ec_bilevel], patch_artist=True)
    ax1.set_title('Energy Consumption')
    ax1.set_ylabel('Value')
    ax1.set_xticklabels(['NSGA-II', 'Bi-level'])

    ax2.boxplot([ld_nsga_ii, ld_bilevel], patch_artist=True)
    ax2.set_title('Average Load Degree')
    ax2.set_ylabel('Value')
    ax2.set_xticklabels(['NSGA-II', 'Bi-level'])
    plt.savefig(os.path.join(ROOT_DIR, f'box_plot'))
    plt.close()

    # Convergence plot NSGA-II
    list_pareto_optimal = []
    optimal_nsga_ii = np.array([])

    # Get nadir
    nadir = [0, 0]
    for f_exp in nsga_ii:
        for i in range(NUMBER_OF_OBJECTIVES):
            nadir[i] = max(nadir[i], np.array(f_exp[0])[:, i].max())
    ind = HV(ref_point=np.array(nadir) + 0.5)
    
    n_iters = len(nsga_ii[0])
    hpl_iters = [[] for _ in range(n_iters)]
    for f_exp in nsga_ii:
        for i, f in enumerate(f_exp):
            front = non_dominated_sorting(f)[0]
            hpl_iters[i].append(ind(np.array(f[front])))

    lower_bound = []
    upper_bound = []
    median = []
    hpl_iters = np.array(hpl_iters)
    for i in range(n_iters):
        lower_bound.append(hpl_iters[i, :].min())
        upper_bound.append(hpl_iters[i, :].max())
        median.append(np.median(hpl_iters[i, :]))

    sns.lineplot(
        x=range(n_iters),
        y=lower_bound,
        c='red',
        linestyle='--'
    )
    sns.lineplot(
        x=range(n_iters),
        y=upper_bound,
        c='red',
        linestyle='--'
    )
    sns.lineplot(
        x=range(n_iters),
        y=median,
        label='Median'
    )

    plt.title('NSGA-II hypervolume convergence plot')
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, f'Convergence_NSGA_II'))
    plt.close()

    # Convergence for Bi-level
    bilevel = np.array(bilevel)
    n_iters = int(len(bilevel[0]))
    ec_iters = [[] for _ in range(n_iters)]
    ld_iters = [[] for _ in range(n_iters)]
    for f_exp in bilevel:
        for i, f in enumerate(f_exp):
            ec_iters[i] = ec_iters[i] + list(f[:, 0])
            ld_iters[i] = ld_iters[i] + list(f[:, 1])

    for i in range(n_iters):
        ec_iters[i] = [np.min(ec_iters[i]), np.median(ec_iters[i]), np.max(ec_iters[i])]
        ld_iters[i] = [np.min(ld_iters[i]), np.median(ld_iters[i]), np.max(ld_iters[i])]
    ec_iters = np.array(ec_iters)
    ld_iters = np.array(ld_iters)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    iter_split = int(n_iters / 2)
    sns.lineplot(
        x=range(n_iters),
        y=ec_iters[:, 0],
        c='red',
        linestyle='--',
        ax=ax1
    )
    sns.lineplot(
        x=range(n_iters),
        y=ec_iters[:, 2],
        c='red',
        linestyle='--',
        ax=ax1
    )
    sns.lineplot(
        x=range(n_iters),
        y=ec_iters[:, 1],
        label='median',
        ax=ax1
    )

    sns.lineplot(
        x=range(iter_split + 1),
        y=ld_iters[iter_split:, 0],
        c='red',
        linestyle='--',
        ax=ax2
    )
    sns.lineplot(
        x=range(iter_split + 1),
        y=ld_iters[iter_split:, 2],
        c='red',
        linestyle='--',
        ax=ax2
    )
    sns.lineplot(
        x=range(iter_split + 1),
        y=ld_iters[iter_split:, 1],
        label='median',
        ax=ax2
    )

    plt.title('Bilevel convergence plot')
    ax1.set_xlabel('Iterations')
    ax2.set_xlabel('Iterations')
    ax1.set_ylabel('Energy Consumption')
    ax2.set_ylabel('Average Load Degree')
    plt.legend()
    plt.savefig(os.path.join(ROOT_DIR, f'Convergence_Bilevel'))
    plt.close()
