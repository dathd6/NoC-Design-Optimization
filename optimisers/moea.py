from moo import MultiObjectiveOptimization

class MOEA(MultiObjectiveOptimization):
    def __init__(self, n_cores, core_graph, mesh_2d_shape):
        super().__init__(n_cores, core_graph, mesh_2d_shape)

    def crossover(self):
        pass

    def mutation(self):
        pass
