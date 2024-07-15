import numpy as np
from time import time
from moo import MultiObjectiveOptimization
from noc import calc_energy_consumption, calc_energy_consumption_with_static_mapping_sequence, calc_load_balance, calc_load_balance_with_static_mapping_sequence, get_router_mappings
from optimisers.bayesian import BayesianOptimization
from optimisers.genetic_algorithm import mutation_heuristic_routing, partially_mapped_crossover, single_swap_mutation_route,single_swap_mutation, sorting_solution, tournament_selection, two_point_crossover
from optimisers.nsga_ii import elitism_replacement, calc_crowding_distance, tournament_selection as nsga_ii_tournament_selection
from utils import record_fitnesses, record_others, record_population

class Bilevel(MultiObjectiveOptimization):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=[], fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

    def optimize_upper_level_BO(self, folder_name, n_iterations=100):
        bo = BayesianOptimization(
            mesh_2d_shape=(self.n_rows, self.n_cols),
            n_cores=self.n_cores,
            es_bit=self.es_bit,
            el_bit=self.el_bit,
            core_graph=self.core_graph,
            population=self.population,
            fitnesses=self.f
        )

        return bo.optimize(folder_name, n_iterations)

    def optimize_upper_level_GA(self, folder_name, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            population = []

            sorting_solution(self.f, self.population)
            record_fitnesses(folder_name, 'lower_level_GA', self.n_iters,self.f.reshape(-1, 1))

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

        record_fitnesses(folder_name, 'upper_level_GA', self.n_iters, self.f.reshape(-1, 1))
        record_others(folder_name, 'upper_level_BO_execution_time', opt_time)
        record_population(folder_name, 'upper_level_optimal_GA', [self.population[0]], n_objectives=1)
        
        return opt_time, self.population[0], self.f[0]

    def optimize_lower_level(self, folder_name, mapping_seq, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            population = []

            sorting_solution(self.f, self.population)

            record_fitnesses(folder_name, 'lower_level_GA', self.n_iters,self.f.reshape(-1, 1))

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

        record_fitnesses(folder_name, 'lower_level_GA', self.n_iters,self.f.reshape(-1, 1))
        record_others(folder_name, 'lower_level_execution_time', opt_time)
        record_population(folder_name, 'lower_level_optimal', [self.population[0]], n_objectives=1)

        return opt_time, self.population[0], self.f[0]

    def optimize_lower_level_moo(self, folder_name, mapping_seq, tournament_size, n_iterations=100):
        opt_time = 0

        while self.n_iters < n_iterations:
            start_time = time()
            self.non_dominated_sorting()
            self.crowding_distance = calc_crowding_distance(
                population=self.population,
                fitnesses=self.f,
                pareto_fronts=self.pareto_fronts
            )
            record_fitnesses(folder_name, 'lower_level_nsga_ii', self.n_iters, self.f)
            population = []
            while len(self.population) + len(population) < 2 * self.size_p:
                parent_a, parent_b = nsga_ii_tournament_selection(
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

            self.non_dominated_sorting()
            self.crowding_distance = calc_crowding_distance(population=self.population, fitnesses=self.f, pareto_fronts=self.pareto_fronts)
            elitism_replacement(self.population, self.f, self.pareto_fronts, self.size_p, self.crowding_distance)

            opt_time += (time() - start_time)
            print(f'Bilevel lower level - NSGA-II Iteration: {self.n_iters + 1} - Time: {opt_time}')
            self.n_iters += 1

        self.non_dominated_sorting()
        self.crowding_distance = calc_crowding_distance(population=self.population, fitnesses=self.f, pareto_fronts=self.pareto_fronts)

        record_fitnesses(folder_name, 'lower_level_nsga_ii', self.n_iters, self.f)
        record_others(folder_name, 'lower_level_nsga_ii_execution_time', opt_time)
        PFs = []
        for i in self.pareto_fronts[0]:
            PFs.append(self.population[i])
        record_population(folder_name, 'lower_levell_nsga_ii_optimal', PFs, n_objectives=1)


        return opt_time, self.f[self.pareto_fronts[0]]
