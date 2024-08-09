import numpy as np
from core.mutation import mutation_heuristic_routing

from problem.noc import random_core_mapping, \
                        random_shortest_routing

def initialize_random_mapping_sequences(n_solutions, n_cores, n_rows, n_cols):
    mapping_seqs = []
    for _ in range(n_solutions):
        mapping_seq = random_core_mapping(n_cores, n_rows, n_cols)
        mapping_seqs.append(mapping_seq)
    mapping_seqs = np.array(mapping_seqs)
    return mapping_seqs

def initialize_random_route(n_solutions, core_graph, n_rows, n_cols, mapping_seq):
    route_paths = []
    for _ in range(n_solutions):
        routes = random_shortest_routing(
            core_graph,
            mapping_seq,
            n_cols
        )
        for _ in range(len(routes) * len(routes)):
            routes = mutation_heuristic_routing(
                parent=routes,
                core_graph=core_graph,
                n_rows=n_rows,
                n_cols=n_cols,
                mapping_seq=mapping_seq,
                rate=1)
        route_paths.append(routes)
    return route_paths

def initialize_random_shortest_route(mapping_seqs, core_graph, n_cols):
    route_paths = []
    for mapping_seq in mapping_seqs:
        routes = random_shortest_routing(
            core_graph,
            mapping_seq,
            n_cols
        )
        route_paths.append(routes)
    return route_paths
