import numpy as np
from fitness_functions import energy_consumption, load_balancing

class Mesh_2D:
    def __init__(self, n_rows, n_cols) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        # Initialize a 2D mesh with no core mapping to router
        self.mesh_2D = np.array([-1] * (n_rows * n_cols)).reshape(n_rows, n_cols)
        
    def random_core_mapping(self, n_cores):
        mapping_order = list(range(n_cores)) + [-1] * (n_cores - self.n_rows * self.n_cols)
        np.random.shuffle(mapping_order)
        self.mesh_2D = np.array(mapping_order).reshape(self.n_rows, self.n_cols)

    def fitness(self):
        self.energy_consumption = energy_consumption(mapping_seqs=new_solution),
        self.load_balancing = load_balancing(mapping_seqs=new_solution)
