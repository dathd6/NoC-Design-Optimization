import numpy as np
from moo import MultiObjectiveOptimization
from pymoo.util.ref_dirs import get_reference_directions

from constants import NUMBER_OF_OBJECTIVES


class MOEAD(MultiObjectiveOptimization):
    def __init__(self, n_cores, core_graph, mesh_2d_shape, n_partitions=10, n_neighbours=3):
        super().__init__(n_cores, core_graph, mesh_2d_shape)
        self.weights = get_reference_directions(
            "uniform",
            NUMBER_OF_OBJECTIVES,
            n_partitions=n_partitions
        )
        self.n_weights = len(self.weights)
        self.T_ = n_neighbours
        self.solutions = np.array([
            solution.get_fitness() for solution in self.population
        ])

        self.z = self.init_z()
        self.b = self.generate_closest_weight_vectors()
        self.nearest_weight = [-1] * self.size_p
        self.w_solutions = self.init_objective_nearest_weight_vector()
