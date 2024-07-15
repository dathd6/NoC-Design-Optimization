import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def euclidean_distance_from_point_to_vector(point, start, end):
    point = np.array(point)
    start = np.array(start)
    end = np.array(end)

    # Calculate the vector representing the line
    vector = end - start

    distance = np.inf
    # Calculate the parameters A, B, and C for the equation of the line
    A = -vector[1]
    B = vector[0]
    C = -(A * start[0] + B * start[1])

    # Calculate the distance using the formula
    distance = np.abs(A * point[0] + B * point[1] + C) / np.sqrt(A**2 + B**2)

    return distance

def find_shortest_route_direction(x1, x2):
    """
        x-axis:
            -1 when the direction is from right to left
            1 when the direction is from left to right
            0 when two router is on the same row
        y-axis:
            -1 when the direction is from bottom to top
            1 when the direction is from top to bottom
            0 when two router is on the same column
    """
    if x1 < x2:
        return 1
    elif x1 > x2:
        return -1
    else:
        return 0

def visualise_perf(filename, optimisers, labels):
    for i, opt in enumerate(optimisers):
        perf_metrics = np.array(opt.perf_metrics)
        sns.lineplot(
            x=perf_metrics[:, 0],
            y=perf_metrics[:, 1],
            label=labels[i],
        )
    plt.xlabel('No. evaluations')
    plt.ylabel('Hypervolume')
    plt.title('HV convergence')
    plt.savefig(filename)
    plt.close()
    # plt.show()

def router_index_to_coordinates(idx, n_cols):
    x = idx // n_cols
    y = idx % n_cols
    return (int(x), int(y))


def core_modification_new_routes(core_graph, modified_cores, core_mapping_coord, n_cols, routes):
    # Change routing path for core that have been swapped
    for i in range(len(core_graph)):
        src, des, _ = core_graph[i]
        if modified_cores.get(src) or modified_cores.get(des):
            r1_x, r1_y = router_index_to_coordinates(core_mapping_coord[src], n_cols)
            r2_x, r2_y = router_index_to_coordinates(core_mapping_coord[des], n_cols)

            new_route = [0] * abs(r1_x - r2_x) + [1] * abs(r1_y - r2_y)
            np.random.shuffle(new_route)
            routes[i] = new_route

def get_file_name(path):
    token = os.path.basename(path).split('.')
    return token[0], token[1]

def get_number_of_cores(core_graph):
    n_cores = 0
    for src, des, _ in core_graph:
        n_cores = np.max([src, des, n_cores])
    return n_cores + 1

def get_latest_test_case(directory_path):
    try:
        subfolders = [f.name for f in os.scandir(directory_path) if f.is_dir()]
        testcase_number = -1
        for subfolder in subfolders:
            testcase_number = np.max([int(subfolder.split('_')[-1]), testcase_number])
        return testcase_number + 1
    except FileNotFoundError:
        return f"Directory {directory_path} not found."
    except Exception:
        return 0

def count_dir(directory, optimiser=None):
    # List all items in the given directory
    items = os.listdir(directory)
    folder_count = 0
    for item in items:
        if optimiser and optimiser in item:
            # Count only directories
            folder_count = folder_count + 1

        if not optimiser:
            folder_count = folder_count + 1

    return folder_count

def dominated(fitness1, fitness2):
    if fitness1[0] <= fitness2[0] and fitness1[1] <= fitness2[1]:
        return True
    return False

def not_equal(fitness1, fitness2):
    if fitness1[0] != fitness2[0] and fitness1[1] != fitness2[1]:
        return True
    return False

def non_dominated_sorting(fitnesses):
    """Fast non-dominated sorting to get list Pareto Fronts"""
    dominating_sets = []
    dominated_counts = []

    # For each solution:
    # - Get solution index that dominated by current solution
    # - Count number of solution dominated current solution
    for f1 in fitnesses:
        current_dominating_set = set()
        dominated_counts.append(0)
        for i, f2 in enumerate(fitnesses):
            if dominated(f1, f2) and not_equal(f1, f2):
                current_dominating_set.add(i)
            elif dominated(f2, f1) and not_equal(f1, f2):
                dominated_counts[-1] += 1
        dominating_sets.append(current_dominating_set)

    dominated_counts = np.array(dominated_counts)
    pareto_fronts = []

    # Append all the pareto fronts and stop when there is no solution being dominated (domintead count = 0)
    while True:
        current_front = np.where(dominated_counts==0)[0]
        if len(current_front) == 0:
            break
        pareto_fronts.append(current_front)
        for individual in current_front:
            dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so will not find it anymore
            dominated_by_current_set = dominating_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                dominated_counts[dominated_by_current] -= 1

    return pareto_fronts

def visualize_objective_space(filename, list_pareto_fronts, fitnesses, title, figsize, labels, alpha=1):
    for opt, pareto_fronts in list_pareto_fronts.items():
        front = pareto_fronts[0]
        non_dominated = np.array([
            solution for solution in fitnesses[opt][front]
        ])
        dominated = []
        for i in range(1, len(pareto_fronts)):
            dominated = dominated + [solution for solution in fitnesses[opt][pareto_fronts[i]]]
        dominated = np.array(dominated)
        if dominated.size != 0:
            sns.scatterplot(
                x=dominated[:, 0],
                y=dominated[:, 1],
                label=f'{opt} dominated',
                alpha=alpha
            )
        sns.scatterplot(
            x=non_dominated[:, 0],
            y=non_dominated[:, 1],
            label=f'{opt} non-dominated',
        )
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig(filename)

def record_others(record_folder, filename, recording):
    with open(f'{record_folder}/{filename}.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(str(recording))

def record_fitnesses(record_folder, filename, iteration, fitnesses):
    with open(f'{record_folder}/{filename}_fitness_{iteration}.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for fitness in fitnesses:
            writer.writerow(fitness)

def record_population(record_folder, filename, population, n_objectives=1):
    if n_objectives == 1:
        with open(f'{record_folder}/{filename}.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for solution in population:
                writer.writerow(solution)

    if n_objectives == 2:
        with open(f'{record_folder}/{filename}_mapping.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for solution in population:
                writer.writerow(solution[0])

        with open(f'{record_folder}/{filename}_route.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for solution in population:
                writer.writerows(solution[1])


