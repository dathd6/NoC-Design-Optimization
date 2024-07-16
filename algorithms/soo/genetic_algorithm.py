import numpy as np
from time import time

from algorithms.base import BaseOptimiser
from core.selection import tournament_selection
from core.sorting import single_objective_sorting
from problem.noc import calc_energy_consumption, calc_load_balance

class GeneticAlgorithm(BaseOptimiser):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=[], fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

    def optimize(self, folder_name, filename, n_iterations, tournament_size, crossover, mutation, objective, mapping_seq=None):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            population = []

            single_objective_sorting(self.f, self.population)
            self.record(folder_name, filename, opt_time, self.f.reshape(-1, 1), [self.population[0]], n_variables=1)

            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = tournament_selection(self.population, size_t=tournament_size)
                child_c, child_d = crossover(parent_a, parent_b)
                population.append(mutation(child_c))
                population.append(mutation(child_d))

            self.population = self.population + population
            self.f = np.append(
                self.f, 
                calc_energy_consumption(
                    mapping_seqs=np.array(population),
                    n_cols=self.n_cols,
                    core_graph=self.core_graph,
                    es_bit=self.es_bit,
                    el_bit=self.el_bit
                ) if objective == 'EC' else
                calc_load_balance(
                    n_cols=self.n_cols,
                    n_rows=self.n_rows,
                    mapping_seqs=np.array([list(mapping_seq)] * self.size_p),
                    route_paths=population,
                    core_graph=self.core_graph
                )
            )

            single_objective_sorting(self.f, self.population)
            # Elitism replacement
            p = []
            for i in range(self.size_p): 
                p.append(self.population[i])
            self.population = p
            self.f = self.f[:self.size_p]

            # Save execution time
            opt_time += (time() - start_time)
            print(f'\r\tGenetic Algorithm Iteration: {self.n_iters + 1} - Time: {opt_time}s', end='')
            self.n_iters += 1

        self.record(folder_name, filename, opt_time, self.f.reshape(-1, 1), [self.population[0]], n_variables=1)

        print('\n')
        
        return opt_time, self.population[0], self.f[0]
