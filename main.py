import os
from argparse import ArgumentParser
from moo import MultiObjectiveOptimization
from noc import calc_energy_consumption_with_static_mapping_sequence, calc_load_balance, calc_load_balance_with_static_mapping_sequence, get_router_mappings
from optimisers.bilevel import Bilevel
import numpy as np
from optimisers.nsga_ii import NSGA_II

from utils import get_file_name, get_latest_test_case, get_number_of_cores
from constants import EXPERIMENTS_DIR

if __name__ == "__main__":
    # Setup input option
    parser = ArgumentParser(description="Network-on-Chips")
    parser.add_argument('--core-graph', type=str, required=True, help='Core/Task graph')
    parser.add_argument('--rows', type=int, required=True, help='Topology rows')
    parser.add_argument('--columns', type=int, required=True, help='Topology columns')
    parser.add_argument('--energy-link-consumed', type=int, default=20, help='Energy consumed when send one bit through a link')
    parser.add_argument('--energy-switch-consumed', type=int, default=30, help='Energy consumed when send one bit through a node')
    parser.add_argument('--experiments', type=int, default=10, help='Number of experiments')
    parser.add_argument('--population', type=int, default=100, help='Number of population')
    parser.add_argument('--tournament', type=int, default=20, help='Number of solutions in the tournament')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    args = parser.parse_args()

    # Get input option
    data = np.loadtxt(args.core_graph, dtype=int)
    n_rows = args.rows
    n_cols = args.columns
    es_bit = args.energy_switch_consumed
    el_bit = args.energy_link_consumed
    n_cores = get_number_of_cores(data)
    n_solutions = args.population
    n_tournaments = args.tournament
    n_iterations = args.iterations

    # Get experiments test case folder
    filename, filetype = get_file_name(args.core_graph)
    if not os.path.exists(EXPERIMENTS_DIR):
        os.mkdir(EXPERIMENTS_DIR)
    ROOT_DIR = os.path.join(EXPERIMENTS_DIR, filename)
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)

    # Start experiments
    for _ in range(args.experiments):
        # Create experiment folder
        TEST_CASE_DIR = os.path.join(ROOT_DIR, f'experiment_{str(get_latest_test_case(ROOT_DIR))}')
        if not os.path.exists(TEST_CASE_DIR):
            os.mkdir(TEST_CASE_DIR)

        # Create initial population
        moo = MultiObjectiveOptimization(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
        )
        moo.intialize_random_mapping_sequences(n_solutions=n_solutions)
        moo.intialize_shortest_routing_task(moo.mapping_seqs)
        moo.evaluation()
        
        print('Finished initialize the population')

        population = []
        for i in range(moo.size_p):
            population.append([moo.mapping_seqs[i], moo.route_paths[i]])

        bil_bo_nsga_ii = Bilevel(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
            population=moo.mapping_seqs.copy(),
            fitnesses=moo.f[:, 0].copy()
        )
        upper_opt_time0, upper_optimal_mapping_seq0, upper_fitness0 = bil_bo_nsga_ii.optimize_upper_level_BO(folder_name=TEST_CASE_DIR, n_iterations=100)
        bil_bo_nsga_ii.population = bil_bo_nsga_ii.initialize_routing_task(n_solutions=n_solutions, mapping_seq=upper_optimal_mapping_seq0)

        f1 = calc_energy_consumption_with_static_mapping_sequence(
            routing_paths=bil_bo_nsga_ii.population,
            es_bit=es_bit,
            el_bit=el_bit,
        ).reshape(-1, 1)
        f2 = calc_load_balance_with_static_mapping_sequence(
            n_rows=n_rows,
            n_cols=n_cols,
            mapping_seq=upper_optimal_mapping_seq0,
            route_paths=bil_bo_nsga_ii.population,
            core_graph=data,
        ).reshape(-1, 1)
        bil_bo_nsga_ii.f = np.concatenate((f1, f2), axis=1)
        lower_opt_time0, lower_f0 = bil_bo_nsga_ii.optimize_lower_level_moo(
            folder_name=TEST_CASE_DIR,
            mapping_seq=upper_optimal_mapping_seq0,
            n_iterations=1000,
            tournament_size=20
        )

        nsga_ii = NSGA_II(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
            population=population,
            fitnesses=moo.f.copy()
        )
        nsga_ii.optimize(folder_name=TEST_CASE_DIR, tournament_size=20)

        bil_bo = Bilevel(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
            population=moo.mapping_seqs.copy(),
            fitnesses=moo.f[:, 0].copy()
        )
        upper_opt_time1, upper_optimal_mapping_seq1, upper_fitness1 = bil_bo.optimize_upper_level_BO(folder_name=TEST_CASE_DIR, n_iterations=50)
        mapping_seqs = np.array(n_solutions * list(upper_optimal_mapping_seq1.reshape(1, -1)))
        bil_bo.population = bil_bo.intialize_shortest_routing_task(mapping_seqs)
        bil_bo.f = calc_load_balance(
            n_rows=moo.n_rows,
            n_cols=moo.n_cols,
            mapping_seqs=mapping_seqs,
            route_paths=bil_bo.population,
            core_graph=moo.core_graph,
        )
        lower_opt_time1, lower_optimal_route1, lower_fitness1 = bil_bo.optimize_lower_level(
            folder_name=TEST_CASE_DIR,
            mapping_seq=upper_optimal_mapping_seq1,
            n_iterations=1000,
            tournament_size=20
        )

        bil_ga = Bilevel(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
            population=moo.mapping_seqs.copy(),
            fitnesses=moo.f[:, 0].copy()
        )
        upper_opt_time2, upper_optimal_mapping_seq2, upper_fitness2 = bil_ga.optimize_upper_level_GA(folder_name=TEST_CASE_DIR, n_iterations=1000, tournament_size=n_tournaments)
        mapping_seqs = np.array(n_solutions * list(upper_optimal_mapping_seq2.reshape(1, -1)))
        bil_ga.population = bil_ga.intialize_shortest_routing_task(mapping_seqs)
        bil_ga.f = calc_load_balance(
            n_rows=moo.n_rows,
            n_cols=moo.n_cols,
            mapping_seqs=mapping_seqs,
            route_paths=bil_ga.population,
            core_graph=moo.core_graph,
        )
        lower_opt_time2, lower_optimal_route2, lower_fitness2 = bil_ga.optimize_lower_level(
            folder_name=TEST_CASE_DIR,
            mapping_seq=upper_optimal_mapping_seq2,
            n_iterations=1000,
            tournament_size=20
        )
        print('END!!!!!!!!!!!')
