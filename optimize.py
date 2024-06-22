import os
from argparse import ArgumentParser
from moo import MultiObjectiveOptimization
from optimisers.nsga_ii import NSGA_II
from optimisers.bilevel import Bilevel
import numpy as np

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
        moo = MultiObjectiveOptimization()
        moo.intialize_population(
            n_solutions=n_solutions,
            n_cores=n_cores,
            es_bit=es_bit,
            el_bit=el_bit,
            mesh_2d_shape=(n_rows, n_cols),
            core_graph=data,
        )
        population = moo.population.copy()

        # For every optimiser run optimization
        optimisers = [
            ('NSGA-II', NSGA_II), 
            ('Bi-level', Bilevel)
        ]
        for name, optimiser in optimisers:
            record_folder = os.path.join(TEST_CASE_DIR, name)
            opt = optimiser(
                record_folder=record_folder,
                n_tournaments=n_tournaments,
                population=population
            )
            opt.optimize(n_iterations=n_iterations)
