from mesh_2d import Mesh2D
from moo import MultiObjectiveOptimization

class MOEA(MultiObjectiveOptimization):
    def __init__(self, n_cores, core_graph, mesh_2d_shape):
        super().__init__(n_cores, core_graph, mesh_2d_shape)

    def mutation(self, solution: Mesh2D):
        pass

    def crossover(self, solution_1: Mesh2D, solution_2: Mesh2D):
        pass
