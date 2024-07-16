import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_performance_metric(self):
    """Calculate hypervolume to the reference point"""
    front = self.pareto_fronts[0]
    solutions = np.array([solution.get_fitness(is_flag=False) for solution in self.population[front]])
    self.perf_metrics.append(
        [self.n_iters, self.ind(solutions)]
    )

def visualise_perf(filename, optimisers, labels):
    for i, opt in enumerate(optimisers):
        perf_metrics = np.array(opt.perf_metrics)
        sns.lineplot(
            x=perf_metrics[:, 0],
            y=perf_metrics[:, 1],
            label=labels[i],
        )
    plt.xlabel('No. evaluations')
    plt.ylabel('Hypervolume')
    plt.title('HV convergence')
    plt.savefig(filename)
    plt.close()


def visualize_objective_space(filename, list_pareto_fronts, fitnesses, title, figsize, labels, alpha=1):
    for opt, pareto_fronts in list_pareto_fronts.items():
        front = pareto_fronts[0]
        non_dominated = np.array([
            solution for solution in fitnesses[opt][front]
        ])
        dominated = []
        for i in range(1, len(pareto_fronts)):
            dominated = dominated + [solution for solution in fitnesses[opt][pareto_fronts[i]]]
        dominated = np.array(dominated)
        if dominated.size != 0:
            sns.scatterplot(
                x=dominated[:, 0],
                y=dominated[:, 1],
                label=f'{opt} dominated',
                alpha=alpha
            )
        sns.scatterplot(
            x=non_dominated[:, 0],
            y=non_dominated[:, 1],
            label=f'{opt} non-dominated',
        )
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(filename)



