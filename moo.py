import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

from constants import NUMBER_OF_OBJECTIVES
from noc import NetworkOnChip

class MultiObjectiveOptimization:
    def __init__(self):
        self.eval_count = 0
        self.population = np.array([])
        self.pareto_fronts = []
        self.perf_metrics = []

    def intialize_population(self, n_solutions, n_cores, mesh_2d_shape, es_bit, el_bit, core_graph):
        n_rows, n_cols = mesh_2d_shape
        population = []
        self.size_p = n_solutions
        for _ in range(n_solutions):
            solution = NetworkOnChip(
                n_cores=n_cores,
                n_rows=n_rows,
                n_cols=n_cols,
                es_bit=es_bit,
                el_bit=el_bit,
                core_graph=core_graph
            )
            population.append(solution)
        self.population = np.array(population)
        # Get 
        ref_point = self.get_nadir()
        self.ind = HV(ref_point=ref_point + 0.5)
            
    def visualize_objective_space(self, title, figsize, labels, c_dominated='#2377B4', c_non_dominated='#FA8010', alpha=1):
        front = self.pareto_fronts[0]
        non_dominated = np.array([
            solution.get_fitness() for solution in self.population[front]
        ])
        dominated = []
        for i in range(1, len(self.pareto_fronts)):
            dominated = dominated + [solution.get_fitness() for solution in self.population[self.pareto_fronts[i]]]
        dominated = np.array(dominated)
        ax = None
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)

        ax = fig.add_subplot(121)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if dominated.size != 0:
            sns.scatterplot(
                x=dominated[:, 0],
                y=dominated[:, 1],
                ax=ax,
                label='dominated',
                color=c_dominated,
                alpha=alpha
            )
        sns.scatterplot(
            x=non_dominated[:, 0],
            y=non_dominated[:, 1],
            ax=ax,
            color=c_non_dominated,
            label='non-dominated',
        )

    def non_dominated_sorting(self):
        """Fast non-dominated sorting to get list Pareto Fronts"""
        dominating_sets = []
        dominated_counts = []

        # For each solution:
        # - Get solution index that dominated by current solution
        # - Count number of solution dominated current solution
        for solution_1 in self.population:
            current_dominating_set = set()
            dominated_counts.append(0)
            for i, solution_2 in enumerate(self.population):
                if solution_1 >= solution_2 and not solution_1 == solution_2:
                    current_dominating_set.add(i)
                elif solution_2 >= solution_1 and not solution_2 == solution_1:
                    dominated_counts[-1] += 1
            dominating_sets.append(current_dominating_set)

        dominated_counts = np.array(dominated_counts)
        self.pareto_fronts = []

        # Append all the pareto fronts and stop when there is no solution being dominated (domintead count = 0)
        while True:
            current_front = np.where(dominated_counts==0)[0]
            if len(current_front) == 0:
                break
            self.pareto_fronts.append(current_front)
            for individual in current_front:
                dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so will not find it anymore
                dominated_by_current_set = dominating_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    dominated_counts[dominated_by_current] -= 1

    def get_nadir(self):
        fitnesses = np.array([solution.get_fitness() for solution in self.population])
        nadir = []
        for i in range(NUMBER_OF_OBJECTIVES):
            nadir.append(fitnesses[:, i].max())
        return np.array(nadir)

    def calc_performance_metric(self):
        """Calculate hypervolume to the reference point"""
        front = self.pareto_fronts[0]
        solutions = np.array([solution.get_fitness() for solution in self.population[front]])
        self.perf_metrics.append(
            [self.eval_count, self.ind(solutions)]
        )

    def optimize(self):
        pass
