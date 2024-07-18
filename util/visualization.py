import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
import seaborn as sns

from core.sorting import non_dominated_sorting

def calc_performance_metric(self):
    """Calculate hypervolume to the reference point"""
    front = self.pareto_fronts[0]
    solutions = np.array([solution.get_fitness(is_flag=False) for solution in self.population[front]])
    self.perf_metrics.append(
        [self.n_iters, self.ind(solutions)]
    )

def visualise_convergence_plot(filename, fitnesses, colors, labels, indicator, title, figsize):
    plt.figure(figsize=figsize)
    for k, f in enumerate(fitnesses):
        n_iters = len(f[0])
        hpl_iters = [[] for _ in range(n_iters)]

        for f_exp in f:
            for i, fitness in enumerate(f_exp):
                front = non_dominated_sorting(fitness)[0]
                hpl_iters[i].append(indicator(np.array(fitness[front])))
        
        lower_bound = []
        upper_bound = []
        median = []
        hpl_iters = np.array(hpl_iters)
        for i in range(n_iters):
            lower_bound.append(hpl_iters[i, :].min())
            upper_bound.append(hpl_iters[i, :].max())
            median.append(np.median(hpl_iters[i, :]))
        
        plt.plot(
            range(n_iters),
            lower_bound,
            color=colors[k],
            alpha=.5,
        )
        plt.plot(
            range(n_iters),
            upper_bound,
            color=colors[k],
            alpha=.5,
        )
        plt.plot(
            range(n_iters),
            median,
            # markers[k],
            label=labels[k],
            c=colors[k],
        )
        plt.fill_between(range(n_iters), lower_bound, upper_bound, color=colors[k], alpha=0.3)
        
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def visualize_objective_space(filename, PFs, f, title, figsize):
    plt.figure(figsize=figsize)

    front = PFs[0]
    non_dominated = f[front]
    dominated = np.array([])

    for i in range(1, len(PFs)):
        if len(dominated) == 0:
            dominated = f[PFs[i]]
        dominated = np.append(dominated, f[PFs[i]], axis=0)
    dominated = np.array(dominated)
    if dominated.size != 0:
        plt.scatter(
            x=dominated[:, 0],
            y=dominated[:, 1],
            label=f'dominated solution',
        )
    plt.scatter(
        x=non_dominated[:, 0],
        y=non_dominated[:, 1],
        label=f'non-dominated solution',
    )
    
    plt.title(title)
    plt.xlabel('Energy Consumption')
    plt.ylabel('Load Balance')
    plt.legend()
    plt.savefig(filename)
    plt.close()
