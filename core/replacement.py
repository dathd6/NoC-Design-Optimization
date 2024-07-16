import numpy as np

def elitism_replacement(population, fitnesses, pareto_fronts, size_p, crowding_distance):
    indices = []
    
    i = 0
    while len(pareto_fronts) != i and len(pareto_fronts[i]) + len(indices) <= size_p:
        for j in pareto_fronts[i]:
            indices.append(j)
        i += 1

    if len(pareto_fronts) != i:
        front = pareto_fronts[i]
        ranking_index = front[np.argsort(crowding_distance[front])]
        current_pop_len = len(population)
        for index in ranking_index[current_pop_len:size_p]:
            indices.append(index)

    p = []
    for i in indices:
        p.append(population[i])
    population = p
    fitnesses = fitnesses[indices]
