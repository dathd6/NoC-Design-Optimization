import numpy as np

def tournament_selection(population, size_t):
    tournament_idx = np.array([True] * size_t + [False] * (len(population) - size_t))
    results = []
    for _ in range(2):
        np.random.shuffle(tournament_idx)
        winner = None
        for i, t in enumerate(tournament_idx):
            if t:
                winner = population[i]
                break
        results.append(winner)
    return (v for v in results)

def tournament_selection_moo(tournament_size, pareto_fronts, crowding_distance, population):
    tournament = np.array([True] * tournament_size + [False] * (len(population) - tournament_size))
    results = []
    for _ in range(2):
        np.random.shuffle(tournament)
        front = []
        for f in pareto_fronts:
            front = []
            for index in f:
                if tournament[index] == 1:
                    front.append(index)
            if len(front) > 0:
                break
        max_index = np.argmax(crowding_distance[front])
        results.append(population[front[max_index]])
    return (v for v in results)
