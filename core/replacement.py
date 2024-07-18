import numpy as np

def elitism_replacement(pareto_fronts, size_p, crowding_distance):
    # pareto_front_percentage = 0.2
    # desired_num_solutions = int(size_p * pareto_front_percentage)

    indices = []
    
    i = 0
    while len(pareto_fronts) != i and len(pareto_fronts[i]) + len(indices) <= size_p:
        front = pareto_fronts[i]
        ranking_index = front[np.argsort(-crowding_distance[front])]

        for index in ranking_index:
            if crowding_distance[index] != 0:
                indices.append(index)

        i += 1

    if len(pareto_fronts) != i:
        front = pareto_fronts[i]
        ranking_index = front[np.argsort(-crowding_distance[front])]
        for index in ranking_index:
            if len(indices) == size_p:
                break
            if crowding_distance[index] != 0:
                indices.append(index)

    return np.array(indices)
