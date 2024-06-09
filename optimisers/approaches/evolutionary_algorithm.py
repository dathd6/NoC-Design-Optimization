import copy
import numpy as np
from constants import CORE_MAPPING_MUTATION_RATE, ROUTING_MUTATION_RATE
from mesh_2d import Mesh2D
from moo import MultiObjectiveOptimization

class MOEA(MultiObjectiveOptimization):
    def __init__(self, n_cores, core_graph, mesh_2d_shape):
        super().__init__(n_cores, core_graph, mesh_2d_shape)

    def single_swap(self, solution):
        gene1, gene2 = np.random.choice(len(solution), size=2, replace=False)
        solution[gene1], solution[gene2] = solution[gene2], solution[gene1]
        return solution

    def mutation(self, solution: Mesh2D):
        mapping_order = copy.copy(solution.mapping_order)
        if np.random.rand() < CORE_MAPPING_MUTATION_RATE:
            # SINGLE SWAP MUTATION
            mapping_order = self.single_swap(mapping_order)
        if np.random.rand() < ROUTING_MUTATION_RATE:
            pass 
        
    def crossover(self, solution_1: Mesh2D, solution_2: Mesh2D):
        pass
