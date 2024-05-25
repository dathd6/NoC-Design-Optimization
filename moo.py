import numpy as np
from mesh_2d import Mesh_2D

class MultiObjectiveOptimization:
    def __init__(self, n_cores, core_graph, mesh_2d_shape):
        self.n_cores = n_cores
        self.core_graph = np.zeros((n_cores, n_cores))
        for row in core_graph:
            self.core_graph[row[0], row[1]] = row[2]
        self.n_rows, self.n_cols = mesh_2d_shape
        self.population = []

    def intialize_population(self, n_solutions):
        for _ in range(n_solutions):
            solution = Mesh_2D(self.n_rows, self.n_cols)
            solution.random_core_mapping(self.n_cores)
            self.population.append(solution)

    def optimize(self):
        pass
