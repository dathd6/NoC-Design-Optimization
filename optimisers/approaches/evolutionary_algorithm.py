import numpy as np
from constants import CORE_MAPPING_CROSSOVER_RATE, CORE_MAPPING_MUTATION_RATE, ROUTING_MUTATION_RATE
from noc import NetworkOnChip
from moo import MultiObjectiveOptimization
from utils import find_shortest_route_direction, get_task_throughput, swap_gene

class MOEA(MultiObjectiveOptimization):
    def __init__(self):
        super().__init__()

    def mutation(self, solution: NetworkOnChip):
        mapping_seq = solution.mapping_seq.copy()
        routes = solution.routes.copy()
        bw_throughput = solution.bw_throughput.copy()
        task_count = solution.task_count.copy()

        if np.random.rand() < CORE_MAPPING_MUTATION_RATE:
            # SINGLE SWAP MUTATION
            gene1, gene2 = np.random.choice(len(mapping_seq), size=2, replace=False)
            mapping_seq[gene1], mapping_seq[gene2] = mapping_seq[gene2], mapping_seq[gene1]

            # Change core mapping coordinate
            core_mapping_coord = solution.core_mapping_coord.copy()
            core_mapping_coord[mapping_seq[gene1]] = gene1
            core_mapping_coord[mapping_seq[gene2]] = gene2

            # Change routing path for core that have been swapped
            for i in range(len(solution.core_graph)):
                src, des, bw = solution.core_graph[i]
                # Get x, y coordinate of each router in 2D mesh
                r1_x, r1_y = solution.router_index_to_coordinates(solution.core_mapping_coord[src])
                r2_x, r2_y = solution.router_index_to_coordinates(solution.core_mapping_coord[des])
                # Get the direction of x and y when finding the shortest route from router 1 to router 2
                x_dir = find_shortest_route_direction(r1_x, r2_x)
                y_dir = find_shortest_route_direction(r1_y, r2_y)

                for core in [mapping_seq[gene1], mapping_seq[gene2]]:
                    if core == src or core == des:
                        new_x, new_y = solution.router_index_to_coordinates(core_mapping_coord[core])
                        # Remove the previous route
                        b, t = get_task_throughput(
                            n_routers=len(bw_throughput),
                            start=(r1_x, r1_y),
                            n_cols=solution.n_cols,
                            dir=(x_dir, y_dir),
                            route=solution.routes[i],
                            bw=bw
                        )
                        bw_throughput = bw_throughput - b
                        task_count = task_count - t

                        if core == src:
                            new_x_dir = find_shortest_route_direction(new_x, r2_x)
                            new_y_dir = find_shortest_route_direction(new_y, r2_y)

                            # Add new generated route
                            # Generate random route (number of feasible routes is |x1 - x2| * |y1 - y2|)
                            route = [0] * abs(new_x - r2_x) + [1] * abs(new_y - r2_y)
                            np.random.shuffle(route)

                            # Add new route
                            b, t = get_task_throughput(
                                n_routers=len(bw_throughput),
                                start=(new_x, new_y),
                                n_cols=solution.n_cols,
                                dir=(new_x_dir, new_y_dir),
                                route=route,
                                bw=bw
                            )
                            bw_throughput = bw_throughput + b
                            task_count = task_count + t

                        elif core == des:
                            new_x_dir = find_shortest_route_direction(r1_x, new_x)
                            new_y_dir = find_shortest_route_direction(r1_y, new_y)

                            # Add new generated route
                            # Generate random route (number of feasible routes is |x1 - x2| * |y1 - y2|)
                            route = [0] * abs(new_x - r1_x) + [1] * abs(new_y - r1_y)
                            np.random.shuffle(route)

                            # Add new route
                            b, t = get_task_throughput(
                                n_routers=len(bw_throughput),
                                start=(r1_x, r1_y),
                                n_cols=solution.n_cols,
                                dir=(new_x_dir, new_y_dir),
                                route=route,
                                bw=bw
                            )
                            bw_throughput = bw_throughput + b
                            task_count = task_count + t


        if np.random.rand() < ROUTING_MUTATION_RATE:
            # Change a random route for a single task
            gene = np.random.randint(len(routes))
            src, des, bw = solution.core_graph[gene]
            # Get x, y coordinate of each router in 2D mesh
            r1_x, r1_y = solution.router_index_to_coordinates(solution.core_mapping_coord[src])
            r2_x, r2_y = solution.router_index_to_coordinates(solution.core_mapping_coord[des])
            # Get the direction of x and y when finding the shortest route from router 1 to router 2
            x_dir = find_shortest_route_direction(r1_x, r2_x)
            y_dir = find_shortest_route_direction(r1_y, r2_y)

            # Remove the previous route
            b, t = get_task_throughput(
                n_routers=len(bw_throughput),
                start=(r1_x, r1_y),
                n_cols=solution.n_cols,
                dir=(x_dir, y_dir),
                route=solution.routes[gene],
                bw=bw
            )
            bw_throughput = bw_throughput - b
            task_count = task_count - t

            # Add new generated route
            # Generate random route (number of feasible routes is |x1 - x2| * |y1 - y2|)
            route = [0] * abs(r1_x - r2_x) + [1] * abs(r1_y - r2_y)
            np.random.shuffle(route)
            b, t = get_task_throughput(
                n_routers=len(bw_throughput),
                start=(r1_x, r1_y),
                n_cols=solution.n_cols,
                dir=(x_dir, y_dir),
                route=route,
                bw=bw
            )
            bw_throughput = bw_throughput + b
            task_count = task_count + t

            # Change new route
            routes[gene] = route

        return NetworkOnChip(
            n_cores=solution.n_cores,
            n_rows=solution.n_rows,
            n_cols=solution.n_cols,
            es_bit=solution.es_bit,
            el_bit=solution.el_bit,
            core_graph=solution.core_graph,
            mapping_seq=mapping_seq,
            routes=routes,
            bw_throughput=bw_throughput,
            task_count=task_count
        )
            
        
    def crossover(self, solution_1: NetworkOnChip, solution_2: NetworkOnChip):
        child_seq_1, child_seq_2 = solution_1.mapping_seq.copy(), solution_2.mapping_seq.copy()
        # Mesh 2D constant attributes
        n_cores = solution_1.n_cores
        n_rows = solution_1.n_rows
        n_cols = solution_1.n_cols
        n_routers = n_rows * n_cols
        es_bit = solution_1.es_bit
        el_bit = solution_1.el_bit
        core_graph = solution_1.core_graph

        def partially_mapped_crossover(parent_a, parent_b):
            # PMX method ~ Partially matched crossover
            child_c = parent_a.copy()
            child_d = parent_b.copy()
            first_point_subset = np.random.randint(1, n_routers - 1)
            second_point_subset = np.random.randint(first_point_subset, n_routers - 1) + 1
            for i in range(first_point_subset, second_point_subset):
                # Partially mapped crossover (PMX)
                swap_gene(child_c, child_c.index(parent_b[i]), i)
                swap_gene(child_d, child_d.index(parent_a[i]), i)
                # cross data
                child_c[i] = parent_b[i]
                child_c[child_c.index(child_c[i])] = parent_a[i]
                # cross data
                child_d[i] = parent_a[i]
                child_d[child_d.index(child_d[i])] = parent_b[i]
            return child_c, child_d

        if np.random.rand() < CORE_MAPPING_CROSSOVER_RATE:
            child_seq_1, child_seq_2 = partially_mapped_crossover(solution_1.mapping_seq, solution_2.mapping_seq) 

        return NetworkOnChip(
            n_cores=n_cores,
            n_rows=n_rows,
            n_cols=n_cols,
            es_bit=es_bit,
            el_bit=el_bit,
            core_graph=core_graph,
            mapping_seq=child_seq_1,
        ), NetworkOnChip(
            n_cores=n_cores,
            n_rows=n_rows,
            n_cols=n_cols,
            es_bit=es_bit,
            el_bit=el_bit,
            core_graph=core_graph,
            mapping_seq=child_seq_2,
        )
