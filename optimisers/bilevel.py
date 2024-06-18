import numpy as np
from constants import NUMBER_OF_OBJECTIVES
from noc import NetworkOnChip
from optimisers.approaches.evolutionary_algorithm import MOEA

class Bilevel(MOEA):
    def __init__(self, population=np.array([]), n_tournaments=10):
        super().__init__(population)
        self.size_t = n_tournaments

    def sorting_solution(self):
        sorted_indices = np.argsort([noc.get_fitness()[0] for noc in self.population])
        self.population = self.population[sorted_indices]

    def tournament_selection(self):
        tournament_idx = np.array([True] * self.size_t + [False] * (len(self.population) - self.size_t))
        results = []
        for _ in range(2):
            np.random.shuffle(tournament_idx)
            tournament = self.population[tournament_idx]
            results.append(tournament[0])
        return results

    def elitism_replacement(self):
        self.population = self.population[:self.size_p]

    def optimize(self, n_evaluations=100):
        is_break = False
        for i in range(NUMBER_OF_OBJECTIVES):
            flag = [True, True]
            flag[i] = True
            flag[np.abs(i - 1)] = False
            for noc in self.population:
                noc.set_flag(flag)

            while True:
                self.sorting_solution()
                if i == 0:
                    self.non_dominated_sorting()
                    self.calc_performance_metric()
                while len(self.population) < 2 * self.size_p:
                    parents = self.tournament_selection()
                    childrens = self.crossover(parents[0], parents[1], flag=flag)
                    self.population = np.append(self.population, [self.mutation(childrens[0], flag=flag)])
                    self.population = np.append(self.population, [self.mutation(childrens[1], flag=flag)])

                    self.eval_count += 1
                    print(f'\tBilevel {i} Evaluation: ', self.eval_count)

                    self.sorting_solution()
                    if i == 0:
                        self.non_dominated_sorting()
                        self.calc_performance_metric()

                    if self.eval_count == n_evaluations:
                        is_break = True
                        break
                    
                self.elitism_replacement()
                if i == 0:
                    self.non_dominated_sorting()
                    self.calc_performance_metric()

                if is_break:
                    break

            if i == 0:
                # Generate new population with best solution for energy consumption (static mapping sequence)
                population = []
                mapping_seq = self.population[0].mapping_seq
                for noc in self.population:
                    population.append(
                        NetworkOnChip(
                            n_cores=noc.n_cores,
                            n_rows=noc.n_rows,
                            n_cols=noc.n_cols,
                            es_bit=noc.es_bit,
                            el_bit=noc.el_bit,
                            core_graph=noc.core_graph,
                            mapping_seq=mapping_seq
                        )
                    )
                self.population = np.array(population)
