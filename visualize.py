import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from core.sorting import non_dominated_sorting
from argparse import ArgumentParser, BooleanOptionalAction

from util.constants import OUTPUT, ANALYSIS, EXPERIMENTS, FITNESS, EXPERIMENT, NSGA_II, BILEVEL_UPPER, BILEVEL_LOWER
from util.utils import count_files
from util.visualization import generate_video_from_images, visualise_convergence_plot, visualize_box_plot, visualize_objective_space

REPORT = 'report'
ENERGY_CONSUMPTION = 'Energy Consumption'
LOAD_BALANCE = 'Load Balance'

algorithms = [ NSGA_II, BILEVEL_UPPER, BILEVEL_LOWER ]
dict_f = { algo: [] for algo in algorithms }

if __name__ == "__main__":
    # Setup input option
    parser = ArgumentParser(description="Visualization")
    parser.add_argument('--application', type=str, required=True, help='NoC Application')
    parser.add_argument('--color', type=str, default='#EA1000', help='Convergence color')
    parser.add_argument('--objective-space', action=BooleanOptionalAction, type=bool, help='Visualize Objective space')
    parser.add_argument('--box-plot', action=BooleanOptionalAction, type=bool, help='Visualize box plot')
    parser.add_argument('--optimal-fitness', action=BooleanOptionalAction, type=bool, help='Average Optimal fitness')
    parser.add_argument('--algorithm-animation', action=BooleanOptionalAction, type=bool, help='Animation algorithm')
    parser.add_argument('--convergence-plot', action=BooleanOptionalAction, type=bool, help='Plot convergence plot')
    args = parser.parse_args()

    # Get input option
    app = args.application
    color = args.color

    # Get experiments test case folder
    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)
    ROOT_DIR = os.path.join(OUTPUT, app)
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)

    ANALYSIS_DIR = os.path.join(ROOT_DIR, ANALYSIS)
    if not os.path.exists(ANALYSIS_DIR):
        os.mkdir(ANALYSIS_DIR)
    RECORD_DIR = os.path.join(ROOT_DIR, EXPERIMENTS)
    
    N_ITERATIONS = 1000
    # Retrieve recorded value
    print('START RETRIEVE DATA !')
    for algo in algorithms:
        ALGO_DIR = os.path.join(RECORD_DIR, algo)
        N_EXPERIMENTS = np.max(count_files(ALGO_DIR))
        f = []

        for i in range(N_EXPERIMENTS):
            FITNESS_DIR = os.path.join(
                os.path.join(
                    ALGO_DIR,
                    f'{EXPERIMENT}_{i}'
                ), FITNESS)
            n_iters = count_files(FITNESS_DIR)
            f_exp = []
            for j in range(n_iters):
                data_iter = np.loadtxt(
                    os.path.join(
                        FITNESS_DIR,
                        f'{FITNESS}_{j}.txt'
                    )
                )
                f_exp.append(data_iter)
            f.append(f_exp)

        dict_f[algo] = f

    if args.optimal_fitness:
        print("START RECORD OPTIMAL FITNESS!!!")
        for algo in algorithms[:2]:
            data = dict_f[algo]
            ec = 0
            lb = 0
            for f_exp in data:
                ec = ec + f_exp[-1][:, 0].min()
                lb = lb + f_exp[-1][:, 1].min()
            with open(f'{ANALYSIS_DIR}/report.txt', 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow(
                    [ENERGY_CONSUMPTION, LOAD_BALANCE]
                ) 
                writer.writerow(
                    [ec / len(data), lb / len(data)]
                ) 

    if args.algorithm_animation:
        print("START ALGORITHMS ANIMATION !!!")

        SAMPLE = np.random.randint(len(dict_f[BILEVEL_UPPER]))
        f_bilevel = dict_f[BILEVEL_UPPER][SAMPLE]
        f_nsga_ii = dict_f[NSGA_II][SAMPLE]

        FRAME_DIR = os.path.join(ANALYSIS_DIR, 'frame')
        if not os.path.exists(FRAME_DIR):
            os.mkdir(FRAME_DIR)

        # Visual objective spaces for every generations
        for i in range(len(f_bilevel)):
            # Create a figure and two subplots (side by side)
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    
            labels = [BILEVEL_UPPER, NSGA_II]
            for j, f in enumerate([f_bilevel[i], f_nsga_ii[i]]):
                PFs = non_dominated_sorting(f)
    
                front = PFs[0]
                non_dominated = f[front]
    
                dominated = np.array([])
                for k in range(1, len(PFs)):
                    if len(dominated) == 0:
                        dominated = f[PFs[k]]
                    dominated = np.append(dominated, f[PFs[k]], axis=0)
                dominated = np.array(dominated)
    
                if dominated.size != 0:
                    ax[j].scatter(
                        x=dominated[:, 0],
                        y=dominated[:, 1],
                        label=f'dominated solution',
                    )
    
                ax[j].scatter(
                    x=non_dominated[:, 0],
                    y=non_dominated[:, 1],
                    label=f'non-dominated solution',
                )
                ax[j].set_title(f'{labels[j]} - Iteration {i}')
                ax[j].set_xlabel(ENERGY_CONSUMPTION)
                ax[j].set_ylabel(LOAD_BALANCE)
                ax[j].legend()
            # Adjust layout
            plt.tight_layout()
            plt.savefig(os.path.join(FRAME_DIR, f'objective_space_{i}'))
            plt.close()

        generate_video_from_images(FRAME_DIR, f'{ANALYSIS_DIR}/animation.mp4')
    
    if args.objective_space:
        fs = []
        list_PFs = []
        bp = []
        for algo in dict_f.keys():
            if algo == BILEVEL_LOWER:
                continue

            f = np.array([])
            for f_exp in dict_f[algo]:
                new_f = None
                if algo == BILEVEL_UPPER:
                    new_f = f_exp[-1][:1]
                else:
                    new_f = f_exp[-1]
                if len(f) == 0:
                    f = new_f
                else:
                    f = np.concatenate((f, new_f), axis=0)

            fs.append(f)
            list_PFs.append(non_dominated_sorting(f))

            if algo == BILEVEL_UPPER:
                bp.append(f)
            else:
                PFs = non_dominated_sorting(f)
                bp.append(f[PFs[0]])

        visualize_box_plot(
            filename=os.path.join(ANALYSIS_DIR, f'box_plot_{app}'),
            fs=bp,
            labels=['NSGA-II', 'Bi-level'],
            figsize=(6, 6),
        )

        visualize_objective_space(
            filename=os.path.join(ANALYSIS_DIR, f'objective_space_{app}'),
            fs=fs,
            list_PFs=list_PFs,
            labels=['NSGA-II', 'Bi-level'],
            is_non_dominated=[True, False],
            figsize=(6, 6),
        )

    if args.convergence_plot:
        print('START VISUALIZE CONVERGENCE PLOT BI-LEVEL !!!')
        for k, algo in enumerate([BILEVEL_UPPER, BILEVEL_LOWER]):
            f = dict_f[algo]

            f_best = [[] for _ in range(N_ITERATIONS)]

            # Get Nadir
            for f_exp in f:

                for i in range(N_ITERATIONS):
                    if k == 0:
                        max_values = f_exp[0].max(axis=0)
                        f_best[i].append(f_exp[i][:, 0].min() / max_values[0])
                    else:
                        max_values = f_exp[0].max()
                        f_best[i].append(f_exp[i].min())

            visualise_convergence_plot(
                filename=os.path.join(ANALYSIS_DIR, f'convergence_{algo}_{app}'),
                f=f_best,
                color=color,
                label=app,
                title=f'{BILEVEL_UPPER if k == 0 else f"last {BILEVEL_LOWER}"} level convergence',
                figsize=(5, 4),
                xlabel='Iterations',
                ylabel=ENERGY_CONSUMPTION if k == 0 else LOAD_BALANCE,
            )

        if args.convergence_plot:
            print('START VISUALIZE CONVERGENCE PLOT NSGA-II !!!')

            f = dict_f[NSGA_II]
            hpl_iters = [[] for _ in range(N_ITERATIONS + 1)]

            # Get Nadir
            nadir = [0, 0]

            for f_exp in f:
                m_value = f_exp[0].max(axis=0)
                nadir[0] = max(nadir[0], m_value[0])
                nadir[1] = 1

            for f_exp in f:

                ind = HV(ref_point=[1, 1])

                n_iters = len(f[0])

                for i, fi in enumerate(f_exp):
                    front = non_dominated_sorting(fi)[0]
                    hpl_iters[i].append(ind(np.array(fi)[front] / nadir))

            visualise_convergence_plot(
                filename=os.path.join(ANALYSIS_DIR, f'convergence_nsga_ii_{app}'),
                f=hpl_iters,
                color=color,
                label=app,
                title='NSGA-II convergence',
                figsize=(5, 4),
                xlabel='Iterations',
                ylabel='Hypervolume'
            )
