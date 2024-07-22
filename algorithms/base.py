import numpy as np

from util.utils import record_fitnesses, record_population, record_time


class BaseOptimiser:
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        self.n_iters = 0
        self.population = population
        self.size_p = len(population)
        self.f = fitnesses
        self.pareto_fronts = []
        self.perf_metrics = []

        self.n_cores = n_cores
        self.n_rows, self.n_cols = mesh_2d_shape
        self.es_bit = es_bit
        self.el_bit = el_bit
        self.core_graph = core_graph

    def slice_population(self, indices):
        p = []
        for i in indices:
            p.append(self.population[i])
        self.population = p
        self.f = self.f[indices]

    def record(self, folder_name, opt_time, f, population, n_variables):
        record_fitnesses(folder_name, self.n_iters, f)
        record_time(folder_name, opt_time, self.n_iters)
        record_population(folder_name, population, iteration=self.n_iters, n_variables=n_variables)
            
    def optimize(self):
        pass
