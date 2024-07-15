import numpy as np
import noc
from constants import CORE_MAPPING_CROSSOVER_RATE, CORE_MAPPING_MUTATION_RATE, ROUTING_CROSSOVER_RATE, ROUTING_MUTATION_RATE
from utils import find_shortest_route_direction, router_index_to_coordinates

def sorting_solution(fitnesses, population):
    sorted_population = []
    sorted_indices = np.argsort(fitnesses)
    for i in sorted_indices:
        sorted_population.append(population[i])
    fitnesses = fitnesses[sorted_indices]
    population = sorted_population
    return population

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

def get_random_gene(chromosome, value):
    genes = [i for i, v in enumerate(chromosome) if value == v]
    return np.random.choice(genes)

def swap_genes(chromosome, gene1, gene2):
    chromosome[gene1], chromosome[gene2] = chromosome[gene2], chromosome[gene1]

def single_swap_mutation_route(parent, rate=CORE_MAPPING_MUTATION_RATE):
    child = parent.copy()

    if np.random.rand() < rate: # Flag to check if this is a single objective optmization
        gene = np.random.randint(len(child))

        if len(child[gene]) >= 2:
            # SINGLE SWAP MUTATION
            sub_gene1, sub_gene2 = np.random.choice(len(child[gene]), size=2, replace=False)
            child[gene][sub_gene1], child[gene][sub_gene2] = child[gene][sub_gene2], child[gene][sub_gene1]

    return child

def single_swap_mutation(parent, rate=CORE_MAPPING_MUTATION_RATE):
    child = parent.copy()

    if np.random.rand() < rate: # Flag to check if this is a single objective optmization
        # SINGLE SWAP MUTATION
        gene1, gene2 = np.random.choice(len(child), size=2, replace=False)
        child[gene1], child[gene2] = child[gene2], child[gene1]

    return child

def two_point_crossover(parent_a, parent_b, rate=ROUTING_CROSSOVER_RATE):
    child_c, child_d = parent_a.copy(), parent_b.copy()
    l = len(child_c)

    if np.random.rand() < rate:
        first_point_subset = np.random.randint(1, l - 1)
        second_point_subset = np.random.randint(first_point_subset, l - 1) + 1
        for i in range(first_point_subset, second_point_subset):
            child_c[i] = parent_b[i].copy()
            child_d[i] = parent_a[i].copy()

    return child_c, child_d


def partially_mapped_crossover(parent_a, parent_b, rate=CORE_MAPPING_CROSSOVER_RATE):
    child_c, child_d = parent_a.copy(), parent_b.copy()

    l = len(child_c)

    if np.random.rand() < rate:
        first_point_subset = np.random.randint(1, l - 1)
        second_point_subset = np.random.randint(first_point_subset, l - 1) + 1

        for i in range(first_point_subset, second_point_subset):
            swap_genes(
                chromosome=child_c,
                gene1=i,
                gene2=get_random_gene(child_c, parent_b[i])
            )

            swap_genes(
                chromosome=child_d,
                gene1=i,
                gene2=get_random_gene(child_d, parent_a[i])
            )

    return child_c, child_d

def mutation_heuristic_routing(parent, core_graph, n_rows, n_cols, mapping_seq, rate=ROUTING_MUTATION_RATE):
    child = parent.copy()
    if np.random.rand() < rate:
        seq = noc.get_router_mappings([mapping_seq])[0]

        validate = False
        count = 0

        route_idx = np.random.randint(len(child))
        route = child[route_idx]
        src, des, _ = core_graph[route_idx]
        new_route = []
        while not validate or count > 9:
            route_idx = np.random.randint(len(child))
            route = child[route_idx]
            src, des, _ = core_graph[route_idx]
            # get x, y coordinate of each router in 2d mesh
            r1_x, r1_y = router_index_to_coordinates(seq[src], n_cols)
            r2_x, r2_y = router_index_to_coordinates(seq[des], n_cols)
            
            if int(np.abs(r1_x - r2_x) + np.abs(r1_y - r2_y)) + 6 >= len(route):
                validate = True

        if not validate:
            return child
        
        # Get the direction of x and y when finding the shortest route from router 1 to router 2
        point = seq[src]
        feasible_step = [(1, 0), (-1, 0), (0, 1), (0, -1)] 

        gene = np.random.choice(len(route))
        for i in range(gene):
            new_route.append(route[i])
            point += route[i]

        x, y = router_index_to_coordinates(point, n_cols)
        if x > 0 and y > 0 and x < n_rows - 1 and y < n_cols - 1:
            if gene > 0:
                prev_step = 0
            else:
                prev_step = route[gene - 1]
            mutation_step = route[gene]
            new_step = route[gene]
            while prev_step == -new_step or mutation_step == new_step:
                idx = np.random.choice(len(feasible_step))
                chosen = feasible_step[idx]
                new_x = x + chosen[0]
                new_y = y + chosen[1]
                if new_x < 0 or new_x >= n_rows or new_y >= n_cols or new_y < 0:
                    continue
                else:
                    new_step =  chosen[0] * n_cols + chosen[1]
            
            new_route.append(new_step)
            point += new_step
        
        # get x, y coordinate of each router in 2d mesh
        r1_x, r1_y = router_index_to_coordinates(point, n_cols)
        r2_x, r2_y = router_index_to_coordinates(seq[des], n_cols)
        # Get the direction of x and y when finding the shortest route from router 1 to router 2
        x_step = find_shortest_route_direction(r1_x, r2_x) * n_cols
        y_step = find_shortest_route_direction(r1_y, r2_y)

        last_route = [x_step] * abs(r1_x - r2_x) + [y_step] * abs(r1_y - r2_y)
        np.random.shuffle(last_route)

        new_route = new_route + list(last_route)

        val_dict = {}

        start_point = seq[src]
        val_dict[start_point] = True
        for i in range(len(new_route)):
            start_point += new_route[i]
            if val_dict.get(start_point):
                return child
            else:
                val_dict[start_point] = True

        child[route_idx] = np.array(new_route)

    return child

def mutation_multi_objectives(parent, core_graph, n_cols, route_rate=ROUTING_MUTATION_RATE, core_rate=CORE_MAPPING_MUTATION_RATE):
    seq_child = parent[0].copy()
    route_child = parent[1].copy()

    if np.random.rand() < core_rate:
        gene1, gene2 = np.random.choice(len(seq_child), size=2, replace=False)
        seq_child[gene1], seq_child[gene2] = seq_child[gene2], seq_child[gene1]
        changed_cores = {}

        changed_cores[seq_child[gene1]] = True
        changed_cores[seq_child[gene2]] = True

        new_route_child = noc.random_shortest_routing(core_graph, seq_child, n_cols)

        for i in range(len(core_graph)):
            src, des, _ = core_graph[i]
            if changed_cores.get(src) or changed_cores.get(des):
                route_child[i] = new_route_child[i]

    if np.random.rand() < route_rate:
        gene = np.random.randint(len(route_child))

        if len(route_child[gene]) >= 2:
            sub_gene1, sub_gene2 = np.random.choice(len(route_child[gene]), size=2, replace=False)
            route_child[gene][sub_gene1], route_child[gene][sub_gene2] = route_child[gene][sub_gene2], route_child[gene][sub_gene1]

    return [seq_child, route_child]
       
def crossover_multi_objectives(parent_a, parent_b, core_graph, n_cols, rate=ROUTING_CROSSOVER_RATE):
    seq_child_c, seq_child_d = parent_a[0].copy(), parent_b[0].copy()
    route_child_c, route_child_d = parent_a[1].copy(), parent_b[1].copy()

    l = len(seq_child_c)

    if np.random.rand() < rate:
        first_point_subset = np.random.randint(1, l - 1)
        second_point_subset = np.random.randint(first_point_subset, l - 1) + 1

        changed_cores_1 = {}
        changed_cores_2 = {}
        for i in range(first_point_subset, second_point_subset):
            j = get_random_gene(seq_child_c, parent_b[0][i])
            changed_cores_1[seq_child_c[i]] = True
            changed_cores_1[seq_child_c[j]] = True
            swap_genes(
                chromosome=seq_child_c,
                gene1=i,
                gene2=j
            )

            j = get_random_gene(seq_child_d, parent_a[0][i])
            changed_cores_2[seq_child_d[i]] = True
            changed_cores_2[seq_child_d[j]] = True
            swap_genes(
                chromosome=seq_child_d,
                gene1=i,
                gene2=j
            )

        new_route_child_c = noc.random_shortest_routing(core_graph, seq_child_c, n_cols)
        new_route_child_d = noc.random_shortest_routing(core_graph, seq_child_d, n_cols)

        for i in range(len(core_graph)):
            src, des, _ = core_graph[i]
            if changed_cores_1.get(src) or changed_cores_1.get(des):
                route_child_c[i] = new_route_child_c[i]
            if changed_cores_2.get(src) or changed_cores_2.get(des):
                route_child_d[i] = new_route_child_d[i]

    return [seq_child_c, route_child_c], [seq_child_d, route_child_d]
