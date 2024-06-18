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

def get_task_throughput(n_rows, n_cols, start, dir, route, bw):
    x, y = start
    x_dir, y_dir = dir
    n_routers = n_rows * n_cols
    bw_throughput = np.zeros(n_routers)
    task_count = np.zeros(n_routers)
    r = x * n_cols  + y
    bw_throughput[r] = bw_throughput[r] + bw
    task_count[r] = task_count[r] + 1
    for path in route:
        if path == 0:
            x = x + x_dir
        elif path == 1:
            y = y + y_dir 
        r = x * n_cols  + y
        bw_throughput[r] = bw_throughput[r] + bw
        task_count[r] = task_count[r] + 1

    return bw_throughput, task_count

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

def swap_cores(seq, core1, core2):
    if seq[core2] == -1:
        empty_cores = [i for i, value in enumerate(seq) if value == -1]
        core2 = np.random.choice(empty_cores)
    seq[core1], seq[core2] = seq[core2], seq[core1]
    return core1, core2

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
    return (x, y)


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
