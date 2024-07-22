import numpy as np

def dominated(fitness1, fitness2):
    if fitness1[0] <= fitness2[0] and fitness1[1] <= fitness2[1]:
        return True
    return False

def not_equal(fitness1, fitness2):
    if fitness1[0] != fitness2[0] and fitness1[1] != fitness2[1]:
        return True
    return False

def non_dominated_sorting(fitnesses):
    """Fast non-dominated sorting to get list Pareto Fronts"""
    dominating_sets = []
    dominated_counts = []

    # For each solution:
    # - Get solution index that dominated by current solution
    # - Count number of solution dominated current solution
    for f1 in fitnesses:
        current_dominating_set = set()
        dominated_counts.append(0)
        for i, f2 in enumerate(fitnesses):
            if dominated(f1, f2) and not_equal(f1, f2):
                current_dominating_set.add(i)
            elif dominated(f2, f1) and not_equal(f1, f2):
                dominated_counts[-1] += 1
        dominating_sets.append(current_dominating_set)

    dominated_counts = np.array(dominated_counts)
    pareto_fronts = []

    # Append all the pareto fronts and stop when there is no solution being dominated (domintead count = 0)
    while True:
        current_front = np.where(dominated_counts==0)[0]
        if len(current_front) == 0:
            break
        pareto_fronts.append(current_front)
        for individual in current_front:
            dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so will not find it anymore
            dominated_by_current_set = dominating_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                dominated_counts[dominated_by_current] -= 1

    return pareto_fronts
