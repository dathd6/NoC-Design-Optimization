import math
import torch
import sys
import warnings
import numpy as np
from time import time
from sklearn.gaussian_process.kernels import RBF, ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm
from scipy.optimize import minimize

from algorithms.base import BaseOptimiser
# from core.selection import random_sampling_acquisition
from problem.noc import calc_energy_consumption
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP

def EI(x, gaussian_process, f_best):
    mu,sigma = gaussian_process.predict(x.reshape(-1, 1),return_std=True)
    z = (f_best - mu) / sigma
    EI = sigma * z * norm.cdf(z) + sigma * norm.pdf(z)
    return -EI

def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def list_to_representative_number(lst):
    n = len(lst)
    factoradic = 0
    for i in range(n):
        rank = 0
        for j in range(i + 1, n):
            if lst[j] < lst[i]:
                rank += 1
        factoradic += rank * factorial(n - i - 1)
    return factoradic


def precompute_factorials(n):
    factorials = [1] * (n + 1)
    for i in range(2, n + 1):
        factorials[i] = factorials[i - 1] * i
    return factorials

def representative_number_to_list(representative_number, n):
    factorials = precompute_factorials(n - 1)
    elements = list(range(n))
    result = []
    number = representative_number

    for i in range(n - 1, -1, -1):
        factorial_value = factorials[i]
        index = int(number // factorial_value)
        number %= factorial_value
        result.append(elements.pop(index))
    
    return result

class BayesianOptimization(BaseOptimiser):
    def __init__(self, mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population=np.array([]), fitnesses=np.array([])):
        super().__init__(mesh_2d_shape, n_cores, es_bit, el_bit, core_graph, population, fitnesses)

    # Optimize Energy Consumption
    def optimize(self, filename, folder_name, n_iterations, n_samples):
        # comment out the next line to see the warning
        warnings.simplefilter('ignore', category=ConvergenceWarning)

        for i in range(len(self.population)):
            for j in range(len(self.population[i])):
                n = self.n_cores
                if self.population[i][j] == -1:
                    self.population[i][j] = n
                    n = n + 1

        X = np.zeros(self.size_p)
        for i in range(len(self.population)):
            X[i] = list_to_representative_number(self.population[i])
        X = X.reshape(-1, 1)
        f = self.f
        opt_time = 0
        while self.n_iters < n_iterations:
            # Starting time and Iteration
            start_time = time()

            self.record(folder_name, filename, opt_time, self.f.reshape(-1, 1), [self.population[0]], n_variables=1)

            # Normalisation
            f_train = (f - np.min(f))/(np.max(f) - np.min(f))

            # Model training
            kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            X_train = torch.tensor(X, dtype=torch.float64)
            f_train = torch.tensor(f_train.reshape(-1, 1), dtype=torch.float64)
            gp_model = SingleTaskGP(X_train, f_train) # GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_mll(mll)
            # gp_model.fit(X, f_train)

            bounds = torch.tensor([[0], [factorial(self.n_rows * self.n_cols) - 1]], dtype=torch.float64)
            ei = ExpectedImprovement(gp_model, best_f=f_train.min(), maximize=False)
            candidate, acq_value = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=1,
                num_restarts=100,
                raw_samples=10000,
                options={"batch_limit": 5, "maxiter": 200},
            )
            print(acq_value)
            upper = math.floor(candidate.numpy().reshape(-1)[0])
            lower = math.ceil(candidate.numpy().reshape(-1)[0])


            # # Maximise acquisition function (minimise cause acqui = -EI)
            # f_best = np.min(f_train)
            # # x = random_sampling_acquisition(X, EI, gp_model, f_best, n_samples=n_samples).reshape(-1)
            # min_ei = float(sys.maxsize)
            # x_optimal = None
            # x = np.random.randint(factorial(self.n_rows * self.n_cols), size=n_samples)
            # for x0 in x:
            #     results = minimize(EI, x0, args=(gp_model, f_best), bounds=bounds)

            #     if results.fun < min_ei:
            #         min_ei = results.fun
            #         x_optimal = results.x

            for x in [upper, lower]:
                f_new = calc_energy_consumption(
                    mapping_seqs=[representative_number_to_list(x, self.n_rows * self.n_cols)],
                    n_cols=self.n_cols,
                    core_graph=self.core_graph,
                    es_bit=self.es_bit,
                    el_bit=self.el_bit
                )
                print('\n', np.min(f))
                print(f_new)
                X = np.concatenate((X, np.array(x).reshape(1, -1)),axis=0)
                f = np.concatenate((f, f_new),axis=0)

            print(len(f))


            # Save execution time
            opt_time += (time() - start_time)
            print(f'\r\tBayesian Optimization Iteration: {self.n_iters + 1} - Time: {opt_time}s', end='')
            self.n_iters = self.n_iters + 1

        self.record(folder_name, filename, opt_time, self.f.reshape(-1, 1), [self.population[0]], n_variables=1)

        idx = np.argmin(f)

        solution = representative_number_to_list(X[idx][0], self.n_rows * self.n_cols)
        for i in range(len(solution)):
            if solution[i] >= self.n_cores:
                solution[i] = -1

        print('\n')
        return opt_time, np.array(solution), f[idx]
