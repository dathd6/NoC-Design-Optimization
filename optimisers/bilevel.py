import numpy as np
from time import time
from moo import MultiObjectiveOptimization
from noc import calc_energy_consumption, calc_load_balance
from optimisers.bayesian import BayesianOptimization
from optimisers.genetic_algorithm import partially_mapped_crossover, single_swap_mutation_route,single_swap_mutation, sorting_solution, tournament_selection, two_point_crossover

class Bilevel(MultiObjectiveOptimization):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=[], fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

    def optimize_upper_level_BO(self, n_iterations=100):
        bo = BayesianOptimization(
            mesh_2d_shape=(self.n_rows, self.n_cols),
            n_cores=self.n_cores,
            es_bit=self.es_bit,
            el_bit=self.el_bit,
            core_graph=self.core_graph,
            population=self.population,
            fitnesses=self.f
        )

        return bo.optimize(n_iterations)

    def optimize_upper_level_GA(self, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            population = []

            sorting_solution(self.f, self.population)

            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = tournament_selection(self.population, size_t=tournament_size)
                child_c, child_d = partially_mapped_crossover(parent_a, parent_b)
                population.append(single_swap_mutation(child_c))
                population.append(single_swap_mutation(child_d))

            self.population = np.concatenate((self.population, population), axis=0)
            self.f = np.append(
                self.f, 
                calc_energy_consumption(
                    mapping_seqs=np.array(population),
                    n_cols=self.n_cols,
                    core_graph=self.core_graph,
                    es_bit=self.es_bit,
                    el_bit=self.el_bit
                )
            )

            sorting_solution(self.f, self.population)
            # Elitism replacement
            self.population = self.population[:self.size_p]
            self.f = self.f[:self.size_p]

            # Save execution time
            opt_time += (time() - start_time)
            print(f'Bilevel upper level - Genetic Algorithm Iteration: {self.n_iters + 1} - Time: {opt_time}')
            self.n_iters += 1

        return opt_time, self.population[0], self.f[0]

    def optimize_lower_level(self, mapping_seq, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            population = []

            sorting_solution(self.f, self.population)

            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = tournament_selection(self.population, size_t=tournament_size)
                child_c, child_d = two_point_crossover(parent_a, parent_b)

                population.append(single_swap_mutation_route(child_c))
                population.append(single_swap_mutation_route(child_d))

            self.population = self.population + population
            self.f = np.append(
                self.f, 
                calc_load_balance(
                    n_cols=self.n_cols,
                    n_rows=self.n_rows,
                    mapping_seqs=np.array([list(mapping_seq)] * self.size_p),
                    route_paths=population,
                    core_graph=self.core_graph
                )
            )

            sorting_solution(self.f, self.population)
            # Elitism replacement
            p = []
            for i in range(self.size_p): 
                p.append(self.population[i])
            self.population = p
            self.f = self.f[:self.size_p]

            # Save execution time
            opt_time += (time() - start_time)
            print(f'Bilevel lower level - Genetic Algorithm Iteration: {self.n_iters + 1} - Time: {opt_time}')
            self.n_iters += 1

        return opt_time, self.population[0], self.f[0]
