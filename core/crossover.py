import numpy as np

from problem.noc import random_shortest_routing

from util.chromosome import get_random_gene, swap_genes
from util.constants import CROSSOVER_RATE

def two_point_crossover(parent_a, parent_b, rate=CROSSOVER_RATE):
    child_c, child_d = parent_a.copy(), parent_b.copy()
    l = len(child_c)

    if np.random.rand() < rate:
        first_point_subset = np.random.randint(1, l - 1)
        second_point_subset = np.random.randint(first_point_subset, l - 1) + 1
        for i in range(first_point_subset, second_point_subset):
            child_c[i] = parent_b[i].copy()
            child_d[i] = parent_a[i].copy()

    return child_c, child_d

def partially_mapped_crossover(parent_a, parent_b, rate=CROSSOVER_RATE):
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

def crossover_multi_objectives(parent_a, parent_b, core_graph, n_cols, rate=CROSSOVER_RATE):
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

        new_route_child_c = random_shortest_routing(core_graph, seq_child_c, n_cols)
        new_route_child_d = random_shortest_routing(core_graph, seq_child_d, n_cols)

        for i in range(len(core_graph)):
            src, des, _ = core_graph[i]
            if changed_cores_1.get(src) or changed_cores_1.get(des):
                route_child_c[i] = new_route_child_c[i]
            if changed_cores_2.get(src) or changed_cores_2.get(des):
                route_child_d[i] = new_route_child_d[i]

    return [seq_child_c, route_child_c], [seq_child_d, route_child_d]
