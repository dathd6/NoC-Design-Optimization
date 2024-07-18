import os
from argparse import ArgumentParser
import numpy as np

from algorithms.base import BaseOptimiser
from algorithms.moo.bilevel import Bilevel
from algorithms.moo.nsga_ii import NSGA_II
from core.initialization import initialize_random_mapping_sequences, initialize_random_route, initialize_random_shortest_route
from problem.noc import calc_energy_consumption, calc_load_balance, calc_energy_consumption_with_static_mapping_sequence, calc_load_balance_with_static_mapping_sequence
from util.utils import get_file_name, \
                       get_latest_test_case, \
                       get_number_of_cores

EXPERIMENT_STR = 'experiment'
EXPERIMENTS_DIR = 'output/experiments/'

if __name__ == "__main__":
    # Setup input option
    parser = ArgumentParser(description="Network-on-Chips")
    parser.add_argument('--core-graph', type=str, required=True, help='Core/Task graph')
    parser.add_argument('--rows', type=int, required=True, help='Topology rows')
    parser.add_argument('--columns', type=int, required=True, help='Topology columns')
    parser.add_argument('--energy-link-consumed', type=int, default=1, help='Energy consumed when send one bit through a link')
    parser.add_argument('--energy-switch-consumed', type=int, default=2, help='Energy consumed when send one bit through a node')
    parser.add_argument('--experiments', type=int, default=10, help='Number of experiments')
    parser.add_argument('--population', type=int, default=100, help='Number of population')
    parser.add_argument('--tournament', type=int, default=20, help='Number of solutions in the tournament')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    args = parser.parse_args()

    # Get input option
    core_graph = np.loadtxt(args.core_graph, dtype=int)
    n_rows = args.rows
    n_cols = args.columns
    es_bit = args.energy_switch_consumed
    el_bit = args.energy_link_consumed
    n_cores = get_number_of_cores(core_graph)
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
    for i in range(args.experiments):
        # Create experiment folder
        TEST_CASE_DIR = os.path.join(ROOT_DIR, f'{EXPERIMENT_STR}_{str(get_latest_test_case(ROOT_DIR))}')
        if not os.path.exists(TEST_CASE_DIR):
            os.mkdir(TEST_CASE_DIR)

        print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        print('*  1. Initialize the population                   ')
        print(f'*    + No. Experiments:        {args.experiments}') 
        print(f'*    - Iterations:             {n_iterations}') 
        print(f'*    - Mesh Topology:          {n_rows}x{n_cols}') 
        print(f'*    - Test Case:              {filename}.{filetype}') 
        print(f'*    - Tournament Size (GA):   {n_tournaments}') 
        print(f'*    - No. Solutions:          {n_solutions}') 
        print(f'*    - ESbit & ELbit:          {es_bit} & {el_bit}') 
        print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *\n')

        # Create initial population
        mapping_seqs = initialize_random_mapping_sequences(n_solutions=n_solutions, n_cores=n_cores, n_rows=n_rows, n_cols=n_cols)
        route_paths = initialize_random_shortest_route(mapping_seqs=mapping_seqs, core_graph=core_graph, n_cols=n_cols)

        f1 = calc_energy_consumption(
            mapping_seqs=mapping_seqs,
            n_cols=n_cols,
            core_graph=core_graph,
            es_bit=es_bit,
            el_bit=el_bit,
        ).reshape(-1, 1)
        f2 = calc_load_balance(
            n_rows=n_rows,
            n_cols=n_cols,
            mapping_seqs=mapping_seqs,
            route_paths=route_paths,
            core_graph=core_graph,
        ).reshape(-1, 1)
        f = np.concatenate((f1, f2), axis=1) 

        print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        print('* 2. Bilevel                                          *')
        print('*    - Upper Level: Bayesian Optimisation             *')
        print('*    - Lower Level: NSGA-II                           *')
        print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        print(f'Experiments no. {i + 1}\n')

        bil_bo_nsga_ii = Bilevel(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=core_graph,
            population=mapping_seqs.copy(),
            fitnesses=f[:, 0].copy()
        )
        upper_opt_time0, upper_optimal_mapping_seq0, upper_fitness0 = bil_bo_nsga_ii.optimize_upper_level_BO(
            filename='bilevel_upper',
            folder_name=TEST_CASE_DIR,
            n_iterations=int(n_iterations)
        )
        bil_bo_nsga_ii.population = initialize_random_route(
            n_solutions=n_solutions,
            core_graph=core_graph,
            n_rows=n_rows,
            n_cols=n_cols,
            mapping_seq=upper_optimal_mapping_seq0
        )

        f1 = calc_energy_consumption_with_static_mapping_sequence(
            routing_paths=bil_bo_nsga_ii.population,
            core_graph=core_graph,
            es_bit=es_bit,
            el_bit=el_bit,
        ).reshape(-1, 1)
        f2 = calc_load_balance_with_static_mapping_sequence(
            n_rows=n_rows,
            n_cols=n_cols,
            mapping_seq=upper_optimal_mapping_seq0,
            route_paths=bil_bo_nsga_ii.population,
            core_graph=core_graph,
        ).reshape(-1, 1)
        bil_bo_nsga_ii.f = np.concatenate((f1, f2), axis=1)
        lower_opt_time0, lower_f0 = bil_bo_nsga_ii.optimize_lower_level_moo(
            filename='bilevel_lower',
            folder_name=TEST_CASE_DIR,
            mapping_seq=upper_optimal_mapping_seq0,
            n_iterations=n_iterations,
            tournament_size=n_tournaments
        )

        print('* * * * * * * *')
        print('* 3. NSGA-II  *')
        print('* * * * * * * *')
        print(f'Experiments no. {i + 1}\n')

        population = []
        for j in range(len(mapping_seqs)):
            population.append([mapping_seqs[j].copy(), route_paths[j].copy()])

        nsga_ii = NSGA_II(
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=core_graph,
            population=population,
            fitnesses=f.copy()
        )
        nsga_ii.optimize(
            filename='nsga_ii',
            folder_name=TEST_CASE_DIR,
            tournament_size=n_tournaments,
            n_iterations=n_iterations
        )
        print(np.min(nsga_ii.f[:, 0]))
        print(upper_fitness0)

        # print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        # print('* 4. Bilevel                                          *')
        # print('*    - Upper Level: Bayesian Optimisation             *')
        # print('*    - Lower Level: Genetic Algortihm                 *')
        # print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        # print(f'Experiments no. {i + 1}\n')

        # bil_bo_ga = Bilevel(
        #     n_cores=n_cores,
        #     es_bit=es_bit,
        #     el_bit=el_bit,
        #     mesh_2d_shape=(n_rows, n_cols),
        #     core_graph=core_graph,
        #     population=mapping_seqs.copy(),
        #     fitnesses=f[:, 0].copy()
        # )
        # upper_opt_time1, upper_optimal_mapping_seq1, upper_fitness1 = bil_bo_ga.optimize_upper_level_BO(
        #     filename='bilevel_bo_ga_upper',
        #     folder_name=TEST_CASE_DIR,
        #     n_iterations=int(n_iterations / 10)
        # )
        # optimal_mapping_seqs = np.array(n_solutions * [list(upper_optimal_mapping_seq1)])
        # bil_bo_ga.population = initialize_random_shortest_route(
        #     optimal_mapping_seqs,
        #     core_graph,
        #     n_cols
        # )
        # bil_bo_ga.f = calc_load_balance(
        #     n_rows=n_rows,
        #     n_cols=n_cols,
        #     mapping_seqs=optimal_mapping_seqs,
        #     route_paths=bil_bo_ga.population,
        #     core_graph=core_graph,
        # )
        # lower_opt_time1, lower_optimal_route1, lower_fitness1 = bil_bo_ga.optimize_lower_level(
        #     filename='bilevel_bo_ga_lower',
        #     folder_name=TEST_CASE_DIR,
        #     mapping_seq=upper_optimal_mapping_seq1,
        #     n_iterations=n_iterations,
        #     tournament_size=n_tournaments
        # )

        # print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        # print('* 5. Bilevel                                          *')
        # print('*    - Upper Level: Genetic Algorithm                 *')
        # print('*    - Lower Level: Genetic Algortihm                 *')
        # print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        # print(f'Experiments no. {i + 1}\n')

        # bil_ga_ga = Bilevel(
        #     n_cores=n_cores,
        #     es_bit=es_bit,
        #     el_bit=el_bit,
        #     mesh_2d_shape=(n_rows, n_cols),
        #     core_graph=core_graph,
        #     population=mapping_seqs.copy(),
        #     fitnesses=f[:, 0].copy()
        # )
        # upper_opt_time2, upper_optimal_mapping_seq2, upper_fitness2 = bil_ga_ga.optimize_upper_level_GA(
        #     filename='bilevel_ga_ga_upper',
        #     folder_name=TEST_CASE_DIR,
        #     n_iterations=n_iterations,
        #     tournament_size=n_tournaments
        # )
        # optimal_mapping_seqs = np.array(n_solutions * [list(upper_optimal_mapping_seq2)])
        # bil_ga_ga.population = initialize_random_shortest_route(optimal_mapping_seqs, core_graph, n_cols)
        # bil_ga_ga.f = calc_load_balance(
        #     n_rows=n_rows,
        #     n_cols=n_cols,
        #     mapping_seqs=optimal_mapping_seqs,
        #     route_paths=bil_ga_ga.population,
        #     core_graph=core_graph,
        # )
        # lower_opt_time2, lower_optimal_route2, lower_fitness2 = bil_ga_ga.optimize_lower_level(
        #     filename='bilevel_ga_ga_lower',
        #     folder_name=TEST_CASE_DIR,
        #     mapping_seq=upper_optimal_mapping_seq2,
        #     n_iterations=n_iterations,
        #     tournament_size=n_tournaments
        # )

        print('!!!!!!!!!!!!! END !!!!!!!!!!!!')
