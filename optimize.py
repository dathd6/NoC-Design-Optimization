import os
from argparse import ArgumentParser, BooleanOptionalAction
import numpy as np

from algorithms.moo.bilevel import Bilevel
from algorithms.moo.nsga_ii import NSGA_II
from core.initialization import initialize_random_mapping_sequences, initialize_random_shortest_route
from problem.noc import calc_energy_consumption, calc_load_balance
from util.utils import count_files, \
                       get_file_name, \
                       get_number_of_cores, \
                       mkdir, record_population

EXPERIMENT_STR = 'experiment'
OUTPUT_DIR = 'output/'

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
    parser.add_argument('--nsga-ii', action=BooleanOptionalAction, type=bool, help='NSGA-II')
    parser.add_argument('--bi-level', action=BooleanOptionalAction, type=bool, help='Bi-level')
    parser.add_argument('--old-population', action=BooleanOptionalAction, type=bool, help='Use old population')
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
    ROOT_DIR = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)
    EXPERIMENTS_DIR = os.path.join(ROOT_DIR, 'experiments')
    if not os.path.exists(EXPERIMENTS_DIR):
        os.mkdir(EXPERIMENTS_DIR)
    POPULATION_DIR = os.path.join(ROOT_DIR, 'population')
    if not os.path.exists(POPULATION_DIR):
        os.mkdir(POPULATION_DIR)

    # Start experiments
    for i in range(args.experiments):
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

        if args.old_population:
            mapping_seqs = np.loadtxt(os.path.join(POPULATION_DIR, f'population_mapping_{i}.txt'))

            with open(os.path.join(POPULATION_DIR, f'population_route_{i}.txt'), 'r') as file:
                lines = file.readlines()
            all_lists = [[int(x) for x in line.strip().split(' ')] for line in lines]
            route_paths = [all_lists[i:i + len(core_graph)] for i in range(0, len(all_lists), len(core_graph))]
        else:
            # Create initial population
            mapping_seqs = initialize_random_mapping_sequences(n_solutions=n_solutions, n_cores=n_cores, n_rows=n_rows, n_cols=n_cols)
            route_paths = initialize_random_shortest_route(mapping_seqs=mapping_seqs, core_graph=core_graph, n_cols=n_cols)

            population = []
            for j in range(n_solutions):
                population.append([mapping_seqs[j], route_paths[j]])

            iteration = int(count_files(POPULATION_DIR) / 2)
            record_population(
                ROOT_DIR,
                population, 
                iteration=iteration,
                n_variables=2
            )

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
        print('*    - Upper Level: GA                                *')
        print('*    - Lower Level: GA                                *')
        print('* * * * * * * * * * * * * * * * * * * * * * * * * * * *')
        print(f'Experiments no. {i + 1}\n')

        if args.bi_level:
            TMP_DIR = mkdir(os.path.join(EXPERIMENTS_DIR, 'bilevel_upper'))
            TEST_CASE_DIR = mkdir(os.path.join(TMP_DIR, f'experiment_{count_files(TMP_DIR)}'))

            TMP_DIR = mkdir(os.path.join(EXPERIMENTS_DIR, 'bilevel_lower'))
            TEST_CASE_DIR_LOWER = mkdir(os.path.join(TMP_DIR, f'experiment_{count_files(TMP_DIR)}'))

            population = []
            for j in range(len(mapping_seqs)):
                population.append([mapping_seqs[j].copy(), route_paths[j].copy()])
            bil = Bilevel(
                n_cores=n_cores,
                es_bit=es_bit,
                el_bit=el_bit,
                mesh_2d_shape=(n_rows, n_cols),
                core_graph=core_graph,
                population=population,
                fitnesses=f.copy()
            )
            bil.optimize(
                folder_name=TEST_CASE_DIR,
                lower_folder=TEST_CASE_DIR_LOWER,
                tournament_size=n_tournaments,
                upper_iters=int(n_iterations),
                lower_iters=int(n_iterations)
            )

            # TMP_DIR = mkdir(os.path.join(EXPERIMENTS_DIR, 'bilevel_lower'))
            # TEST_CASE_DIR = mkdir(os.path.join(TMP_DIR, f'experiment_{count_files(TMP_DIR)}'))
            # bil_bo_nsga_ii.population = initialize_random_shortest_route(
            #     mapping_seqs=[upper_optimal_mapping_seq0] * n_solutions,
            #     core_graph=core_graph,
            #     n_cols=n_cols,
            # )
            # bil_bo_nsga_ii.f = calc_load_balance(
            #     n_cols=n_cols,
            #     n_rows=n_rows,
            #     route_paths=bil_bo_nsga_ii.population,
            #     mapping_seqs=[upper_optimal_mapping_seq0] * n_solutions,
            #     core_graph=core_graph
            # )
            # bil_bo_nsga_ii.optimize_lower_level(
            #     folder_name=TEST_CASE_DIR,
            #     mapping_seq=upper_optimal_mapping_seq0,
            #     n_iterations=n_iterations,
            #     tournament_size=n_tournaments
            # )

        print('* * * * * * * *')
        print('* 3. NSGA-II  *')
        print('* * * * * * * *')
        print(f'Experiments no. {i + 1}\n')

        if args.nsga_ii:
            TMP_DIR = mkdir(os.path.join(EXPERIMENTS_DIR, 'nsga_ii'))
            TEST_CASE_DIR = mkdir(os.path.join(TMP_DIR, f'experiment_{count_files(TMP_DIR)}'))

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
                folder_name=TEST_CASE_DIR,
                tournament_size=n_tournaments,
                n_iterations=n_iterations
            )

        print('!!!!!!!!!!!!! END !!!!!!!!!!!!')
