import numpy as np
 
from constants import NUMBER_OF_OBJECTIVES
from optimisers.approaches.evolutionary_algorithm import MOEA

class NSGA_II(MOEA):
    def __init__(self, n_cores, core_graph, mesh_2d_shape, n_tournaments=10):
        super().__init__(n_cores, core_graph, mesh_2d_shape)
        self.size_t = n_tournaments

    def calc_crowding_distance(self):
        self.crowding_distance = np.zeros(len(self.population))

        for front in self.pareto_fronts:
            fitnesses = np.array([
                solution.get_fitness() for solution in self.population[front]
            ])
        
            # Normalise each objectives, so they are in the range [0,1]
            # This is necessary, so each objective's contribution have the same magnitude to the crowding distance.
            normalized_fitnesses = np.zeros_like(fitnesses)

            for j in range(NUMBER_OF_OBJECTIVES):
                min_val = np.min(fitnesses[:, j])
                max_val = np.max(fitnesses[:, j])
                val_range = max_val - min_val
                normalized_fitnesses[:, j] = (fitnesses[:, j] - min_val) / val_range

            for j in range(NUMBER_OF_OBJECTIVES):
                idx = np.argsort(fitnesses[:, j])
                
                self.crowding_distance[idx[0]] = np.inf
                self.crowding_distance[idx[-1]] = np.inf
                if len(idx) > 2:
                    for i in range(1, len(idx) - 1):
                        self.crowding_distance[idx[i]] += normalized_fitnesses[idx[i + 1], j] - normalized_fitnesses[idx[i - 1], j]

    def tournament_selection(self):
        tournament = np.array([True] * self.size_t + [False] * (len(self.population) - self.size_t))
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
        return results

    def elitism_replacement(self):
        elitism = self.population.copy()
        population = []
        
        i = 0
        while len(self.pareto_fronts[i]) + len(population) <= self.size_p:
            for solution in elitism[self.pareto_fronts[i]]:
                population.append(solution)
            i += 1

        front = self.pareto_fronts[i]
        ranking_index = front[np.argsort(self.crowding_distance[front])]
        current_pop_len = len(population)
        for index in ranking_index[current_pop_len:self.size_p]:
            population.append(elitism[index])
        self.population = np.array(population)

    def optimize(self):
        is_break = False
        while True:
            self.non_dominated_sorting()
            self.calc_crowding_distance()
            self.calc_performance_metric()
            while len(self.population) < 2 * self.size_p:
                parents = self.tournament_selection()
                childrens = parents[0].crossover(parents[1])
                self.population = np.append(self.population, [self.mutation(childrens[0])])
                self.eval_count += 2
                print('\tEvaluation: ', self.eval_count)
                if self.eval_count == self.n_evaluations:
                    is_break = True
                    self.non_dominated_sorting()
                    self.calc_crowding_distance()
                    break
                self.population = np.append(self.population, [self.mutation(childrens[1])])
                self.eval_count += 1
                print('\tEvaluation: ', self.eval_count)
                if self.eval_count == self.n_evaluations:
                    is_break = True
                    self.non_dominated_sorting()
                    self.calc_crowding_distance()
                    break
                self.non_dominated_sorting()
                self.calc_crowding_distance()

            self.elitism_replacement()

            if is_break:
                break
        self.non_dominated_sorting()
        self.calc_performance_metric()

