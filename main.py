from optimisers.nsga_ii import NSGA_II
from optimisers.bilevel import Bilevel
import numpy as np

from utils import visualise_perf

COREGRAPH = 'dataset/bandwidth/263enc_spare.txt'

if __name__ == "__main__":
    for i in range(10):
        data = np.loadtxt(COREGRAPH, dtype=int)
        nsga_ii = NSGA_II(n_tournaments=40)
        nsga_ii.intialize_population(
            n_solutions=200,
            n_cores=13,
            es_bit=30,
            el_bit=20,
            mesh_2d_shape=(4, 4),
            core_graph=data,
        )
        population = nsga_ii.population.copy()
        nsga_ii.optimize(n_evaluations=1000)

        nsga_ii.visualize_objective_space(
            filename=f'./experiments/NSGA_II_objective_space_{i}',
            title='NSGA-II objective space',
            figsize=(12, 6),
            labels=['Energy consumption', 'Average Load Degree']
        )

        bilevel = Bilevel(
            n_tournaments=40,
            population=population
        )
        bilevel.ind = nsga_ii.ind
        bilevel.optimize(n_evaluations=1000)

        with open(f'./experiments/Bilevel_best_solution_{i}.txt', 'w') as file:
            file.write(str(bilevel.population[0].energy_consumption) + '\n')
            file.write(str(bilevel.population[0].avg_load_degree) + '\n')

        with open(f'./experiments/NSGA_II_best_solution_{i}.txt', 'w') as file:
            for noc in nsga_ii.population[nsga_ii.pareto_fronts[0]]:
                fitness = noc.get_fitness(is_flag=False)
                file.write(str(fitness[0]) + ' ' + str(fitness[1]) + '\n')

        bilevel.visualize_objective_space(
            filename=f'./experiments/Bilevel_objective_space_{i}',
            title='Bilevel objective space',
            figsize=(12, 6),
            labels=['Energy consumption', 'Average Load Degree']
        )

        visualise_perf(
            filename=f'./experiments/convergence_plot_{i}',
            optimisers=[
                nsga_ii,
                bilevel
            ],
            labels=[
                'NSGA-II',
                'Bi-level'
            ]
        )
