import numpy as np
 
from time import time

from pymoo.indicators.hv import HV

from algorithms.base import BaseOptimiser
from core.crossover import crossover_multi_objectives
from core.mutation import mutation_multi_objectives
from core.replacement import elitism_replacement
from core.selection import tournament_selection_moo
from core.sorting import non_dominated_sorting
from problem.noc import calc_energy_consumption, calc_load_balance
from util.population import get_nadir, get_optimal_solutions

def calc_performance_metric(self):
    """Calculate hypervolume to the reference point"""
    front = self.pareto_fronts[0]
    solutions = np.array([solution.get_fitness(is_flag=False) for solution in self.population[front]])
    self.perf_metrics.append(
        [self.n_iters, self.ind(solutions)]
    )

def calc_crowding_distance(fitnesses, pareto_fronts):
    crowding_distance = []

    for front in pareto_fronts:
        f = fitnesses[front]
        cw = np.zeros(len(f))

        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
        normalized_fitnesses = np.zeros_like(f)

        for j in range(2):
            min_val = np.min(f[:, j])
            max_val = np.max(f[:, j])
            val_range = max_val - min_val
            normalized_fitnesses[:, j] = (f[:, j] - min_val) / val_range

        for j in range(2):
            idx = np.argsort(f[:, j])
            
            cw[idx[0]] = np.inf
            cw[idx[-1]] = np.inf
            if len(idx) > 2:
                for i in range(0, len(idx) - 1):
                    cw[idx[i]] = cw[idx[i]] + (normalized_fitnesses[idx[i + 1], j] - normalized_fitnesses[idx[i - 1], j])
        crowding_distance.append(cw)

    return crowding_distance

class NSGA_II(BaseOptimiser):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

        ref_point = get_nadir(self.f)
        self.ind = HV(ref_point=ref_point + 0.5)
        
    def optimize(self, folder_name, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            self.pareto_fronts = non_dominated_sorting(self.f)
            self.crowding_distance = calc_crowding_distance(
                fitnesses=self.f,
                pareto_fronts=self.pareto_fronts
            )
            self.record(folder_name, opt_time, self.f, get_optimal_solutions(self.pareto_fronts, self.population), n_variables=2)
            population = []
            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = tournament_selection_moo(tournament_size, self.pareto_fronts, self.crowding_distance, self.population)
                child_c, child_d = crossover_multi_objectives(parent_a, parent_b, core_graph=self.core_graph, n_cols=self.n_cols)

                new_solution_e = mutation_multi_objectives(child_c, core_graph=self.core_graph, n_cols=self.n_cols)
                new_solution_f = mutation_multi_objectives(child_d, core_graph=self.core_graph, n_cols=self.n_cols)
                population.append(new_solution_e)
                population.append(new_solution_f)

            self.population = self.population + population
            mapping_seqs = []
            route_paths = []
            for solution in population:
                mapping_seqs.append(solution[0])
                route_paths.append(solution[1])
            mapping_seqs = np.array(mapping_seqs)

            f1 = calc_energy_consumption(
                mapping_seqs=mapping_seqs,
                n_cols=self.n_cols,
                core_graph=self.core_graph,
                es_bit=self.es_bit,
                el_bit=self.el_bit,
            ).reshape(-1, 1)
            f2 = calc_load_balance(
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                mapping_seqs=mapping_seqs,
                route_paths=route_paths,
                core_graph=self.core_graph,
            ).reshape(-1, 1)
            f = np.concatenate((f1, f2), axis=1)
            self.f = np.concatenate((self.f, f), axis=0)

            self.pareto_fronts = non_dominated_sorting(self.f)
            self.crowding_distance = calc_crowding_distance(fitnesses=self.f, pareto_fronts=self.pareto_fronts)
            indices = elitism_replacement(self.pareto_fronts, self.size_p, self.crowding_distance)
            self.slice_population(indices)

            opt_time += (time() - start_time)
            print(f'\r\tNSGA-II Iteration: {self.n_iters + 1} - Time: {opt_time}')
            self.n_iters += 1

        self.pareto_fronts = non_dominated_sorting(self.f)
        self.crowding_distance = calc_crowding_distance(fitnesses=self.f, pareto_fronts=self.pareto_fronts)

        self.record(folder_name, opt_time, self.f, get_optimal_solutions(self.pareto_fronts, self.population), n_variables=2)
        print('\n')

        return opt_time, self.f[self.pareto_fronts[0]]
