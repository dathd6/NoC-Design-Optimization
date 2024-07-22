import numpy as np
from time import time

from algorithms.base import BaseOptimiser
from algorithms.moo.nsga_ii import calc_crowding_distance
from algorithms.soo.bayesian import BayesianOptimization
from algorithms.soo.genetic_algorithm import GeneticAlgorithm
from core.crossover import partially_mapped_crossover, two_point_crossover
from core.mutation import mutation_heuristic_routing, single_swap_mutation, single_swap_mutation_sub_gene
from core.replacement import elitism_replacement
from core.selection import tournament_selection_moo
from core.sorting import non_dominated_sorting
from problem.noc import calc_energy_consumption_with_static_mapping_sequence, calc_load_balance_with_static_mapping_sequence
from util.population import get_optimal_solutions

class Bilevel(BaseOptimiser):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=[], fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

    def optimize_upper_level_BO(self, filename, folder_name, n_samples, n_iterations=100):
        bo = BayesianOptimization(
            mesh_2d_shape=(self.n_rows, self.n_cols),
            n_cores=self.n_cores,
            es_bit=self.es_bit,
            el_bit=self.el_bit,
            core_graph=self.core_graph,
            population=self.population,
            fitnesses=self.f
        )

        return bo.optimize(filename, folder_name, n_iterations, n_samples)

    def optimize_upper_level_GA(self, folder_name, tournament_size, n_iterations=100):
        ga = GeneticAlgorithm(
            mesh_2d_shape=(self.n_rows, self.n_cols),
            n_cores=self.n_cores,
            es_bit=self.es_bit,
            el_bit=self.el_bit,
            core_graph=self.core_graph,
            population=self.population,
            fitnesses=self.f
        )

        return ga.optimize(
            folder_name,
            n_iterations,
            tournament_size,
            crossover=partially_mapped_crossover,
            mutation=single_swap_mutation,
            objective='EC'
        )

    def optimize_lower_level(self, filename, folder_name, mapping_seq, tournament_size, n_iterations=100):
        ga = GeneticAlgorithm(
            mesh_2d_shape=(self.n_rows, self.n_cols),
            n_cores=self.n_cores,
            es_bit=self.es_bit,
            el_bit=self.el_bit,
            core_graph=self.core_graph,
            population=self.population,
            fitnesses=self.f
        )

        return ga.optimize(
            folder_name,
            filename,
            n_iterations,
            tournament_size,
            crossover=two_point_crossover,
            mutation=single_swap_mutation_sub_gene,
            objective='LB',
            mapping_seq=mapping_seq
        )

    def optimize_lower_level_moo(self, folder_name, mapping_seq, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            self.pareto_fronts = non_dominated_sorting(self.f)
            self.crowding_distance = calc_crowding_distance(
                fitnesses=self.f,
                pareto_fronts=self.pareto_fronts
            )
            self.record(folder_name, opt_time, self.f,  get_optimal_solutions(self.pareto_fronts, self.population), n_variables=1)
            population = []
            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = tournament_selection_moo(
                    tournament_size=tournament_size,
                    pareto_fronts=self.pareto_fronts,
                    crowding_distance=self.crowding_distance,
                    population=self.population
                )
                child_c, child_d = two_point_crossover(parent_a, parent_b)

                new_solution_e = mutation_heuristic_routing(child_c, core_graph=self.core_graph, n_rows=self.n_rows, n_cols=self.n_cols, mapping_seq=mapping_seq)
                new_solution_f = mutation_heuristic_routing(child_d, core_graph=self.core_graph, n_rows=self.n_rows, n_cols=self.n_cols, mapping_seq=mapping_seq)
                population.append(new_solution_e)
                population.append(new_solution_f)

            self.population = self.population + population

            f1 = calc_energy_consumption_with_static_mapping_sequence(
                routing_paths=population,
                core_graph=self.core_graph,
                es_bit=self.es_bit,
                el_bit=self.el_bit,
            ).reshape(-1, 1)
            f2 = calc_load_balance_with_static_mapping_sequence(
                n_rows=self.n_rows,
                n_cols=self.n_cols,
                mapping_seq=mapping_seq,
                route_paths=population,
                core_graph=self.core_graph,
            ).reshape(-1, 1)
            f = np.concatenate((f1, f2), axis=1)
            self.f = np.concatenate((self.f, f), axis=0)

            self.pareto_fronts = non_dominated_sorting(self.f)
            self.crowding_distance = calc_crowding_distance(fitnesses=self.f, pareto_fronts=self.pareto_fronts)
            indices = elitism_replacement(self.pareto_fronts, self.size_p, self.crowding_distance)
            self.slice_population(indices)

            opt_time += (time() - start_time)
            print(f'\r\tNSGA-II Iteration: {self.n_iters + 1} - Time: {opt_time}s', end='')
            self.n_iters += 1

        self.pareto_fronts = non_dominated_sorting(self.f)
        self.crowding_distance = calc_crowding_distance(fitnesses=self.f, pareto_fronts=self.pareto_fronts)

        self.record(folder_name, opt_time, self.f,  get_optimal_solutions(self.pareto_fronts, self.population), n_variables=1)
        print('\n')

        return opt_time, self.f[self.pareto_fronts[0]]
