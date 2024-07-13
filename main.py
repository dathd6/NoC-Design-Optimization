import os
from argparse import ArgumentParser
from moo import MultiObjectiveOptimization
from noc import calc_load_balance
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
        nsga_ii = NSGA_II(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
            population=population,
            fitnesses=moo.f.copy()
        )
        nsga_ii.optimize(tournament_size=20)

        bil_bo = Bilevel(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
            population=moo.mapping_seqs.copy(),
            fitnesses=moo.f[:, 0].copy()
        )
        upper_opt_time1, upper_optimal_mapping_seq1, upper_fitness1 = bil_bo.optimize_upper_level_BO(n_iterations=50)
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
        upper_opt_time2, upper_optimal_mapping_seq2, upper_fitness2 = bil_ga.optimize_upper_level_GA(n_iterations=1000, tournament_size=n_tournaments)
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
            mapping_seq=upper_optimal_mapping_seq2,
            n_iterations=1000,
            tournament_size=20
        )

        print(upper_opt_time1, upper_optimal_mapping_seq1, upper_fitness1)
        print(lower_opt_time1, lower_optimal_route1, lower_fitness1)

        print(upper_opt_time2, upper_optimal_mapping_seq2, upper_fitness2)
        print(lower_opt_time2, lower_optimal_route2, lower_fitness2)

        # For every optimiser run optimization
        # optimisers = [
        #     ('NSGA-II', NSGA_II), 
        #     ('Bi-level', Bilevel)
        # ]
        # for name, optimiser in optimisers:
        #     record_folder = os.path.join(TEST_CASE_DIR, name)
        #     opt = optimiser(
        #         record_folder=record_folder,
        #         n_tournaments=n_tournaments,
        #         population=population
        #     )
        #     opt.optimize(n_iterations=n_iterations)
