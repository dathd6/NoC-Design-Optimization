import numpy as np
 
from time import time

from pymoo.indicators.hv import HV
from moo import MultiObjectiveOptimization
from noc import calc_energy_consumption, calc_load_balance
from optimisers.genetic_algorithm import crossover_multi_objectives, mutation_multi_objectives, partially_mapped_crossover
from utils import record_fitnesses, record_others, record_population

def calc_crowding_distance(population, fitnesses, pareto_fronts):
    crowding_distance = np.zeros(len(population))

    for front in pareto_fronts:
        fitnesses = fitnesses[front]

        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
        normalized_fitnesses = np.zeros_like(fitnesses)

        for j in range(2):
            min_val = np.min(fitnesses[:, j])
            max_val = np.max(fitnesses[:, j])
            val_range = max_val - min_val
            normalized_fitnesses[:, j] = (fitnesses[:, j] - min_val) / val_range

        for j in range(2):
            idx = np.argsort(fitnesses[:, j])
            
            crowding_distance[idx[0]] = np.inf
            crowding_distance[idx[-1]] = np.inf
            if len(idx) > 2:
                for i in range(1, len(idx) - 1):
                    crowding_distance[idx[i]] += normalized_fitnesses[idx[i + 1], j] - normalized_fitnesses[idx[i - 1], j]

    return crowding_distance

def elitism_replacement(population, fitnesses, pareto_fronts, size_p, crowding_distance):
    indices = []
    
    i = 0
    while len(pareto_fronts) != i and len(pareto_fronts[i]) + len(indices) <= size_p:
        for j in pareto_fronts[i]:
            indices.append(j)
        i += 1

    if len(pareto_fronts) != i:
        front = pareto_fronts[i]
        ranking_index = front[np.argsort(crowding_distance[front])]
        current_pop_len = len(population)
        for index in ranking_index[current_pop_len:size_p]:
            indices.append(index)

    p = []
    for i in indices:
        p.append(population[i])
    population = p
    fitnesses = fitnesses[indices]

def tournament_selection(tournament_size, pareto_fronts, crowding_distance, population):
    tournament = np.array([True] * tournament_size + [False] * (len(population) - tournament_size))
    results = []
    for _ in range(2):
        np.random.shuffle(tournament)
        front = []
        for f in pareto_fronts:
            front = []
            for index in f:
                if tournament[index] == 1:
                    front.append(index)
            if len(front) > 0:
                break
        max_index = np.argmax(crowding_distance[front])
        results.append(population[front[max_index]])
    return (v for v in results)


class NSGA_II(MultiObjectiveOptimization):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

        ref_point = self.get_nadir()
        self.ind = HV(ref_point=ref_point + 0.5)

    def optimize(self, folder_name, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            self.non_dominated_sorting()
            self.crowding_distance = calc_crowding_distance(population=self.population, fitnesses=self.f, pareto_fronts=self.pareto_fronts)
            record_fitnesses(folder_name, 'nsga_ii_fitness', self.n_iters, self.f)
            population = []
            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = tournament_selection(tournament_size, self.pareto_fronts, self.crowding_distance, self.population)
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

            self.non_dominated_sorting()
            self.crowding_distance = calc_crowding_distance(population=self.population, fitnesses=self.f, pareto_fronts=self.pareto_fronts)
            elitism_replacement(self.population, self.f, self.pareto_fronts, self.size_p, self.crowding_distance)

            opt_time += (time() - start_time)
            print(f'NSGA-II Iteration: {self.n_iters + 1} - Time: {opt_time}')
            self.n_iters += 1

        self.non_dominated_sorting()
        self.crowding_distance = calc_crowding_distance(population=self.population, fitnesses=self.f, pareto_fronts=self.pareto_fronts)

        record_fitnesses(folder_name, 'nsga_ii_fitness', self.n_iters, self.f)
        record_others(folder_name, 'nsga_ii_execution_time', opt_time)
        PFs = []
        for i in self.pareto_fronts[0]:
            PFs.append(self.population[i])
        record_population(folder_name, 'NSGA_II_optimal', PFs, n_objectives=1)

        return opt_time, self.f[self.pareto_fronts[0]]
