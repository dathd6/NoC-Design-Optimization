from moo import MultiObjectiveOptimization

class MultiSurrogateBayesian(MultiObjectiveOptimization):
    def __init__(self, n_cores, core_graph, mesh_2d_shape):
        super().__init__(n_cores, core_graph, mesh_2d_shape)
