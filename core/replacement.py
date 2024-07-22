import numpy as np

def elitism_replacement(pareto_fronts, size_p, crowding_distance):
    indices = []
    
    i = 0
    while len(pareto_fronts) != i and len(pareto_fronts[i]) + len(indices) <= size_p:
        front = pareto_fronts[i]
        sorted_indices = np.argsort(-crowding_distance[i])

        for j in sorted_indices:
            indices.append(front[j])

        i += 1

    if len(pareto_fronts) != i:
        front = pareto_fronts[i]
        sorted_indices = np.argsort(-crowding_distance[i])
        for j in sorted_indices:
            indices.append(front[j])

    return np.array(indices)
