import numpy as np

from problem.noc import get_core_mapping_dict,  \
                        router_index_to_coordinates, \
                        random_shortest_routing, \
                        find_shortest_route_direction

from util.constants import MUTATION_RATE

def mutation_multi_objectives(parent, core_graph, n_cols, rate=MUTATION_RATE):
    seq_child = parent[0].copy()
    route_child = parent[1].copy()

    if np.random.rand() < rate:
        gene1, gene2 = np.random.choice(len(seq_child), size=2, replace=False)
        seq_child[gene1], seq_child[gene2] = seq_child[gene2], seq_child[gene1]
        changed_cores = {}

        changed_cores[seq_child[gene1]] = True
        changed_cores[seq_child[gene2]] = True

        new_route_child = random_shortest_routing(core_graph, seq_child, n_cols)

        for i in range(len(core_graph)):
            src, des, _ = core_graph[i]
            if changed_cores.get(src) or changed_cores.get(des):
                route_child[i] = new_route_child[i]

    if np.random.rand() < rate:
        gene = np.random.randint(len(route_child))

        if len(route_child[gene]) >= 2:
            sub_gene1, sub_gene2 = np.random.choice(len(route_child[gene]), size=2, replace=False)
            route_child[gene][sub_gene1], route_child[gene][sub_gene2] = route_child[gene][sub_gene2], route_child[gene][sub_gene1]

    return [seq_child, route_child]

def mutation_heuristic_routing(parent, core_graph, n_rows, n_cols, mapping_seq, rate=MUTATION_RATE):
    child = parent.copy()
    if np.random.rand() < rate:
        seq = get_core_mapping_dict(mapping_seq)

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

def single_swap_mutation_sub_gene(parent, rate=MUTATION_RATE):
    child = parent.copy()

    if np.random.rand() < rate: # Flag to check if this is a single objective optmization
        gene = np.random.randint(len(child))

        if len(child[gene]) >= 2:
            # SINGLE SWAP MUTATION
            sub_gene1, sub_gene2 = np.random.choice(len(child[gene]), size=2, replace=False)
            child[gene][sub_gene1], child[gene][sub_gene2] = child[gene][sub_gene2], child[gene][sub_gene1]

    return child

def single_swap_mutation(parent, rate=MUTATION_RATE):
    child = parent.copy()

    if np.random.rand() < rate: # Flag to check if this is a single objective optmization
        # SINGLE SWAP MUTATION
        gene1, gene2 = np.random.choice(len(child), size=2, replace=False)
        child[gene1], child[gene2] = child[gene2], child[gene1]

    return child
