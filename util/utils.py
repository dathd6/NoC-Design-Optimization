import csv
import os
import numpy as np

def get_number_of_cores(core_graph):
    n_cores = 0
    for src, des, _ in core_graph:
        n_cores = np.max([src, des, n_cores])
    return n_cores + 1

def get_file_name(path):
    token = os.path.basename(path).split('.')
    return token[0], token[1]

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

def count_files(directory, keyword=None, exclude=None):
    # List all items in the given directory
    items = os.listdir(directory)
    folder_count = 0
    for item in items:
        if exclude and exclude in item:
            continue

        if keyword and keyword in item:
            # Count only directories
            folder_count = folder_count + 1

        if not keyword:
            folder_count = folder_count + 1

    return folder_count

def record_time(record_folder, filename, time, iteration):
    folder = os.path.join(record_folder, 'time')
    if not os.path.exists(folder):
        os.mkdir(folder)
    avg_time = 0 if iteration == 0 else time / iteration
    with open(f'{folder}/{filename}_{iteration}.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(f'{time}s - {avg_time}s')

def record_fitnesses(record_folder, filename, iteration, fitnesses):
    folder = os.path.join(record_folder, 'fitness')
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open(f'{folder}/{filename}_{iteration}.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for fitness in fitnesses:
            writer.writerow(fitness)

def record_population(record_folder, filename, population, iteration, n_variables=1):
    folder = os.path.join(record_folder, 'population')
    if not os.path.exists(folder):
        os.mkdir(folder)

    if n_variables == 1:
        with open(f'{folder}/{filename}_{iteration}.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for solution in population:
                writer.writerow(solution)

    if n_variables == 2:
        with open(f'{folder}/{filename}_mapping_{iteration}.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for solution in population:
                writer.writerow(solution[0])

        with open(f'{folder}/{filename}_route_{iteration}.txt', 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for solution in population:
                writer.writerows(solution[1])
