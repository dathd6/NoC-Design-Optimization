import os
import numpy as np
from pymoo.indicators.hv import HV

from argparse import ArgumentParser
from core.sorting import non_dominated_sorting

from util.utils import count_files
from util.visualization import visualize_objective_space, visualise_convergence_plot

ANALYSIS_DIR = 'output/analysis/'
EXPERIMENTS_DIR = 'output/experiments/'

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

    RECORD_DIR = os.path.join(EXPERIMENTS_DIR, testcase_name)
    n_experiments = count_files(RECORD_DIR)
    
    fitness = {
        'nsga_ii': [],
        'bilevel_lower': [],
    }

    for opt_name in fitness.keys():
        for i in range(n_experiments):
            EXPERIMENT_DIR = os.path.join(os.path.join(RECORD_DIR, f'experiment_{i}'), 'fitness')
            items = os.listdir(EXPERIMENT_DIR)
            n_iters = count_files(EXPERIMENT_DIR, opt_name) 
            f_exp = []
            for j in range(n_iters):
                data_iter = np.loadtxt(os.path.join(EXPERIMENT_DIR, f'{opt_name}_{j}.txt'))
                f_exp.append(data_iter)
            fitness[opt_name].append(f_exp)

    # Get Nadir
    nadir = [0, 0]
    for opt_name in fitness.keys():
        for f_exp in fitness[opt_name]:
            for i in range(2):
                nadir[i] = max(nadir[i], np.array(f_exp[0])[:, i].max())
    ind = HV(ref_point=np.array(nadir) + 0.5)

    print('1. NSGA-II: Visualize the objective space')

    f = np.array([])
    for f_exp in fitness['nsga_ii']:
        if len(f) == 0:
            f = f_exp[-1]
        f = np.concatenate((f, f_exp[-1]), axis=0)
    PFs = non_dominated_sorting(f)

    visualize_objective_space(
        filename=os.path.join(ROOT_DIR, 'nsga_ii_objective_space'),
        PFs=PFs,
        f=f,
        title='NSGA-II objective space',
        figsize=(8, 8),
    )

    # visualise_convergence_plot(
    #     filename=os.path.join(ROOT_DIR, 'nsga_ii_convergence_plot'),
    #     f=fitness['nsga_ii'],
    #     indicator=ind,
    #     title='NSGA-II Convergence Plot',
    #     figsize=(10, 8)
    # )

    print('2. Bilevel BO & NSGA-II: visualize the objective space')

    f = np.array([])
    for f_exp in fitness['bilevel_lower']:
        if len(f) == 0:
            f = f_exp[-1]
        f = np.concatenate((f, f_exp[-1]), axis=0)
    PFs = non_dominated_sorting(f)

    visualize_objective_space(
        filename=os.path.join(ROOT_DIR, 'bilevel_lower'),
        PFs=PFs,
        f=f,
        title='Bilevel BO & NSGA-II objective space',
        figsize=(8, 8),
    )

    visualise_convergence_plot(
        filename=os.path.join(ROOT_DIR, 'convergence_plot'),
        fitnesses=[fitness['bilevel_lower'], fitness['nsga_ii']],
        colors=['#7676A1', '#80A176'],
        labels=['Bilevel', 'NSGA-II'],
        # markers=['b-*', 'b-^'],
        indicator=ind,
        title='Convergence Plot',
        figsize=(10, 8)
    )
