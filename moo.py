import numpy as np

from constants import NUMBER_OF_OBJECTIVES
from noc import calc_energy_consumption, calc_load_balance, get_router_mappings, random_core_mapping, random_shortest_routing

def is_dominated(solution1, solution2):
    if solution1[0] == solution1[0] and solution2[1] == solution2[1]:
        return False
    if solution1[0] <= solution2[0] and solution1[1] <= solution2[1]:
        return True
    return False

class MultiObjectiveOptimization:
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        self.n_iters = 0
        self.population = population
        self.f = fitnesses
        self.size_p = len(population)
        self.pareto_fronts = []
        self.perf_metrics = []

        self.n_cores = n_cores
        self.n_rows, self.n_cols = mesh_2d_shape
        self.es_bit = es_bit
        self.el_bit = el_bit
        self.core_graph = core_graph

    def intialize_random_mapping_sequences(self, n_solutions):
        self.size_p = n_solutions
        mapping_seqs = []
        for _ in range(n_solutions):
            mapping_seq = random_core_mapping(self.n_cores, self.n_rows, self.n_cols)
            mapping_seqs.append(mapping_seq)
        self.mapping_seqs = np.array(mapping_seqs)
        return self.mapping_seqs

    def intialize_shortest_routing_task(self, mapping_seqs):
        self.route_paths = []
        for mapping_seq in mapping_seqs:
            routes = random_shortest_routing(self.core_graph, mapping_seq, self.n_cols)
            self.route_paths.append(routes)
        return self.route_paths

    def evaluation(self):
        f1 = calc_energy_consumption(
            mapping_seqs=self.mapping_seqs,
            n_cols=self.n_cols,
            core_graph=self.core_graph,
            es_bit=self.es_bit,
            el_bit=self.el_bit,
        ).reshape(-1, 1)
        f2 = calc_load_balance(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            mapping_seqs=self.mapping_seqs,
            route_paths=self.route_paths,
            core_graph=self.core_graph,
        ).reshape(-1, 1)

        self.f = np.concatenate((f1, f2), axis=1) 
            
    def non_dominated_sorting(self):
        """Fast non-dominated sorting to get list Pareto Fronts"""
        dominating_sets = []
        dominated_counts = []

        # For each solution:
        # - Get solution index that dominated by current solution
        # - Count number of solution dominated current solution
        for solution_1 in self.f:
            current_dominating_set = set()
            dominated_counts.append(0)
            for j, solution_2 in enumerate(self.f):
                if is_dominated(solution_1, solution_2):
                    current_dominating_set.add(j)
                elif is_dominated(solution_2, solution_1):
                    dominated_counts[-1] += 1
            dominating_sets.append(current_dominating_set)

        dominated_counts = np.array(dominated_counts)
        self.pareto_fronts = []

        # Append all the pareto fronts and stop when there is no solution being dominated (domintead count = 0)
        while True:
            current_front = np.where(dominated_counts==0)[0]
            if len(current_front) == 0:
                break
            self.pareto_fronts.append(current_front)
            for individual in current_front:
                dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so will not find it anymore
                dominated_by_current_set = dominating_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    dominated_counts[dominated_by_current] -= 1

    def get_nadir(self):
        nadir = []
        for i in range(NUMBER_OF_OBJECTIVES):
            nadir.append(self.f[:, i].max())
        return np.array(nadir)

    def calc_performance_metric(self):
        """Calculate hypervolume to the reference point"""
        front = self.pareto_fronts[0]
        solutions = np.array([solution.get_fitness(is_flag=False) for solution in self.population[front]])
        self.perf_metrics.append(
            [self.n_iters, self.ind(solutions)]
        )

    def optimize(self):
        pass
