import numpy as np

def get_nadir(f):
    nadir = []
    for i in range(2):
        nadir.append(f[:, i].max())
    return np.array(nadir)

def get_optimal_solutions(pareto_fronts, population):
    PFs = []
    for i in pareto_fronts[0]:
        PFs.append(population[i])
    return PFs

