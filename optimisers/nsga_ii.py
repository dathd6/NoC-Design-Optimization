import numpy as np
 
from time import time

from pymoo.indicators.hv import HV
from moo import MultiObjectiveOptimization
from noc import calc_energy_consumption, calc_load_balance
from optimisers.genetic_algorithm import crossover_multi_objectives, mutation_multi_objectives, partially_mapped_crossover

class NSGA_II(MultiObjectiveOptimization):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

        ref_point = self.get_nadir()
        self.ind = HV(ref_point=ref_point + 0.5)

    def calc_crowding_distance(self):
        self.crowding_distance = np.zeros(len(self.population))

        for front in self.pareto_fronts:
            fitnesses = self.f[front]

            # Normalise each objectives, so they are in the range [0,1]
            # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
            normalized_fitnesses = np.zeros_like(fitnesses)

            for j in range(2):
                min_val = np.min(self.f[:, j])
                max_val = np.max(self.f[:, j])
                val_range = max_val - min_val
                normalized_fitnesses[:, j] = (fitnesses[:, j] - min_val) / val_range

            for j in range(2):
                idx = np.argsort(fitnesses[:, j])
                
                self.crowding_distance[idx[0]] = np.inf
                self.crowding_distance[idx[-1]] = np.inf
                if len(idx) > 2:
                    for i in range(1, len(idx) - 1):
                        self.crowding_distance[idx[i]] += normalized_fitnesses[idx[i + 1], j] - normalized_fitnesses[idx[i - 1], j]

    def tournament_selection(self, tournament_size):
        tournament = np.array([True] * tournament_size + [False] * (len(self.population) - tournament_size))
        results = []
        for _ in range(2):
            np.random.shuffle(tournament)
            front = []
            for f in self.pareto_fronts:
                front = []
                for index in f:
                    if tournament[index] == 1:
                        front.append(index)
                if len(front) > 0:
                    break
            max_index = np.argmax(self.crowding_distance[front])
            results.append(self.population[front[max_index]])
        return (v for v in results)

    def elitism_replacement(self):
        elitism = self.population.copy()
        population = []
        f = []
        indices = []
        
        i = 0
        while len(self.pareto_fronts[i]) + len(indices) <= self.size_p:
            for j in self.pareto_fronts[i]:
                indices.append(j)
            i += 1

        front = self.pareto_fronts[i]
        ranking_index = front[np.argsort(self.crowding_distance[front])]
        current_pop_len = len(population)
        for index in ranking_index[current_pop_len:self.size_p]:
            indices.append(index)

        population = []
        for i in indices:
            population.append(self.population[i])
        self.population = population
        self.f = self.f[indices]

    def optimize(self, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            self.non_dominated_sorting()
            self.calc_crowding_distance()
            # self.record_population()
            population = []
            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = self.tournament_selection(tournament_size)
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
            self.calc_crowding_distance()
            self.elitism_replacement()

            opt_time += (time() - start_time)
            print(f'NSGA-II Iteration: {self.n_iters + 1} - Time: {opt_time}')
            self.n_iters += 1

        self.non_dominated_sorting()
        self.calc_crowding_distance()
        # self.record_population()
        return opt_time, self.f[self.pareto_fronts[0]]
