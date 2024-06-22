import numpy as np
from constants import CORE_MAPPING_CROSSOVER_RATE, CORE_MAPPING_MUTATION_RATE, ROUTING_CROSSOVER_RATE, ROUTING_MUTATION_RATE
from noc import NetworkOnChip
from moo import MultiObjectiveOptimization
from utils import core_modification_new_routes, swap_cores

class MOEA(MultiObjectiveOptimization):
    def __init__(self, record_folder=None, population=np.array([])):
        super().__init__(record_folder=record_folder, population=population)

    def mutation(self, solution: NetworkOnChip, flag=[True, True]):
        mapping_seq = solution.mapping_seq.copy()
        routes = solution.routes.copy()
        core_mapping_coord = solution.core_mapping_coord.copy()
        modified_cores = {}

        if flag[0] and np.random.rand() < CORE_MAPPING_MUTATION_RATE: # Flag to check if this is a single objective optmization
            # SINGLE SWAP MUTATION
            gene1, gene2 = np.random.choice(len(mapping_seq), size=2, replace=False)
            mapping_seq[gene1], mapping_seq[gene2] = mapping_seq[gene2], mapping_seq[gene1]
            modified_cores[mapping_seq[gene1]] = True
            modified_cores[mapping_seq[gene2]] = True

            # Change core mapping coordinate
            if mapping_seq[gene1] != -1:
                core_mapping_coord[mapping_seq[gene1]] = gene1
            if mapping_seq[gene2] != -1:
                core_mapping_coord[mapping_seq[gene2]] = gene2

            core_modification_new_routes(
                core_graph=solution.core_graph,
                modified_cores=modified_cores,
                core_mapping_coord=core_mapping_coord,
                n_cols=solution.n_cols,
                routes=routes
            )

        if flag[1] and np.random.rand() < ROUTING_MUTATION_RATE:
            
            n_mutation = np.random.randint(len(routes))
            indices = np.random.choice(len(routes), size=n_mutation, replace=False)

            for idx in indices:
                # SINGLE SWAP MUTATION
                route = routes[idx]

                if len(route) <= 1:
                    continue
                gene1, gene2 = np.random.choice(len(route), size=2, replace=False)
                route[gene1], route[gene2] = route[gene2], route[gene1]

        return NetworkOnChip(
            n_cores=solution.n_cores,
            n_rows=solution.n_rows,
            n_cols=solution.n_cols,
            es_bit=solution.es_bit,
            el_bit=solution.el_bit,
            core_graph=solution.core_graph,
            mapping_seq=mapping_seq,
            routes=routes,
            flag=flag
        )
            
        
    def crossover(self, solution_1: NetworkOnChip, solution_2: NetworkOnChip, flag=[True, True]):
        # Mesh 2D constant attributes
        n_cores = solution_1.n_cores
        n_rows = solution_1.n_rows
        n_cols = solution_1.n_cols
        n_routers = n_rows * n_cols
        es_bit = solution_1.es_bit
        el_bit = solution_1.el_bit
        core_graph = solution_1.core_graph

        # Mapping sequence crossover
        mapping_seq_1, mapping_seq_2 = solution_1.mapping_seq.copy(), solution_2.mapping_seq.copy()
        routes1, routes2 = solution_1.routes.copy(), solution_2.routes.copy()
        core_mapping_coord_1, core_mapping_coord_2 = solution_1.core_mapping_coord.copy(), solution_2.core_mapping_coord.copy()

        if flag[0] and np.random.rand() < CORE_MAPPING_CROSSOVER_RATE:
            first_point_subset = np.random.randint(1, n_routers - 1)
            second_point_subset = np.random.randint(first_point_subset, n_routers - 1) + 1
            modified_cores_1 = {}
            modified_cores_2 = {}
            for i in range(first_point_subset, second_point_subset):
                # Partially mapped crossover (PMX)
                idx1, idx2 = swap_cores(
                    seq=mapping_seq_1,
                    core1=i,
                    core2=mapping_seq_1.index(solution_2.mapping_seq[i]),
                )
                if mapping_seq_1[idx1] != -1:
                    core_mapping_coord_1[mapping_seq_1[idx1]] = idx1
                    modified_cores_1[mapping_seq_1[idx1]] = True
                if mapping_seq_1[idx2] != -1:
                    core_mapping_coord_1[mapping_seq_1[idx2]] = idx2
                    modified_cores_1[mapping_seq_1[idx2]] = True

                idx1, idx2 = swap_cores(
                    seq=mapping_seq_2,
                    core1=i,
                    core2=mapping_seq_2.index(solution_1.mapping_seq[i]),
                )
                if mapping_seq_2[idx1] != -1:
                    core_mapping_coord_2[mapping_seq_2[idx1]] = idx1
                    modified_cores_2[mapping_seq_2[idx1]] = True
                if mapping_seq_2[idx2] != -1:
                    core_mapping_coord_2[mapping_seq_2[idx2]] = idx2
                    modified_cores_2[mapping_seq_2[idx2]] = True

            core_modification_new_routes(
                core_graph=core_graph,
                modified_cores=modified_cores_1,
                core_mapping_coord=core_mapping_coord_1,
                n_cols=n_cols,
                routes=routes1
            )
            core_modification_new_routes(
                core_graph=core_graph,
                modified_cores=modified_cores_2,
                core_mapping_coord=core_mapping_coord_2,
                n_cols=n_cols,
                routes=routes2
            )

        # Route crossover
        if not flag[0] and np.random.rand() < ROUTING_CROSSOVER_RATE:
            # Need the same mapping sequence to crossover the route
            first_point_subset = np.random.randint(1, len(routes1) - 1)
            second_point_subset = np.random.randint(first_point_subset, len(routes1) - 1) + 1
            child_c = routes1.copy()
            child_d = routes2.copy()
            for i in range(first_point_subset, second_point_subset):
                child_c[i] = routes2[i]
                child_d[i] = routes1[i]
            routes1 = child_c
            routes2 = child_d

        return NetworkOnChip(
            n_cores=n_cores,
            n_rows=n_rows,
            n_cols=n_cols,
            es_bit=es_bit,
            el_bit=el_bit,
            core_graph=core_graph,
            mapping_seq=mapping_seq_1,
            routes=routes1,
            flag=flag
        ), NetworkOnChip(
            n_cores=n_cores,
            n_rows=n_rows,
            n_cols=n_cols,
            es_bit=es_bit,
            el_bit=el_bit,
            core_graph=core_graph,
            mapping_seq=mapping_seq_2,
            routes=routes2,
            flag=flag
        )
