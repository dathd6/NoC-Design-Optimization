import numpy as np
from fitness_functions import energy_consumption, load_balancing

class MultiObjectiveOptimization:
    def __init__(self, n_cores, core_graph, mesh_topology):
        self.n_cores = n_cores
        self.core_graph = np.zeros((n_cores, n_cores))
        for row in core_graph:
            self.core_graph[row[0], row[1]] = row[2]
        self.r_topology, self.c_topology = mesh_topology
        self.population = []
        self.fitnesses = []

    def intialize_population(self, n_solutions):
        mapping_order = list(range(self.n_cores)) + [-1] * (self.r_topology * self.c_topology)
        for _ in range(n_solutions):
            np.random.shuffle(mapping_order)
            solution = np.array(mapping_order).reshape(self.r_topology, self.c_topology)
            self.add_solution(solution)

    def add_solution(self, new_solution):
        self.population.append(new_solution)
        self.fitnesses.append((
            energy_consumption(mapping_seqs=new_solution),
            load_balancing(mapping_seqs=new_solution)
        ))

    def optimize(self):
        pass
