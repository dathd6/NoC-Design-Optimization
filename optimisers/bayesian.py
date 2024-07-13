import numpy as np
from time import time
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

from moo import MultiObjectiveOptimization
from noc import calc_energy_consumption
from optimisers.genetic_algorithm import single_swap_mutation, partially_mapped_crossover

def EI(x, gaussian_process, f_best):
    mu,sigma = gaussian_process.predict(x.reshape(1, -1),return_std=True)
    s = (f_best - mu)/sigma
    EI = -(sigma*s * norm.cdf(s) + sigma*norm.pdf(s))
    EI = EI.reshape(-1)
    return EI

class BayesianOptimization(MultiObjectiveOptimization):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

    # Optimize Energy Consumption
    def optimize(self, n_iterations):
        X = self.population
        f = self.f
        opt_time = 0
        while self.n_iters < n_iterations:
            # Starting time and Iteration
            start_time = time()

            # Normalisation
            f_train = (f - np.min(f))/(np.max(f) - np.min(f))

            # Model training
            kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
            gp_model.fit(X, f_train)

            # Maximise acquisition function (minimise cause acqui = -EI)
            f_best = np.min(f)
            x0 = X[np.argmin(f)]
            ei_0 = EI(np.array([x0]), gp_model, f_best)
            x_new = None
            for _ in range(self.n_iters):
                x_sample = X[np.random.randint(len(X))]
                x, _ = partially_mapped_crossover(x0, x_sample)
                x = single_swap_mutation(x)

                # X_new = np.concatenate((X, x.reshape(1, -1)), axis=0)
                ei = EI(np.array(x), gp_model, f_best)
                if ei_0 > ei:
                    ei_0 = ei
                    x0 = x
                    x_new = x

                if np.abs(ei - ei_0) < 1e-6:
                    break

            if not x_new is None:
                f_new = calc_energy_consumption(
                    mapping_seqs=[x_new],
                    n_cols=self.n_cols,
                    core_graph=self.core_graph,
                    es_bit=self.es_bit,
                    el_bit=self.el_bit
                )

                X = np.concatenate((X, x_new.reshape(1, -1)),axis=0)
                f = np.concatenate((f, f_new),axis=0)

            # Save execution time
            opt_time += (time() - start_time)
            print(f'Bilvel upper level - Bayesian Optimization Iteration: {self.n_iters + 1} - Time: {opt_time}s')
            self.n_iters = self.n_iters + 1

        idx = np.argmin(f)
        return opt_time, X[idx], f[idx]
