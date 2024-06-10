import numpy as np

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

def get_task_through_put(n_routers, start, dir, route, bw):
    x, y = start
    x_dir, y_dir = dir

    bw_throughput = np.zeros(n_routers)
    task_count = np.zeros(n_routers)
    for path in route:
        if path == 0:
            x = x + x_dir 
        elif path == 1:
            y = y + y_dir
        r = (x + 1) * (y + 1) - 1
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

def swap_gene(chromosome, gene1=None, gene2=None):
    new_solution = chromosome.copy()
    # choose two random genes
    # If there are none given gene
    if not gene1:
        gene1 = np.random.randint(0, len(new_solution) - 1)
    if not gene2:
        gene2 = np.random.randint(0, len(new_solution)- 1)
    # Swap
    temp = new_solution[gene1]
    new_solution[gene1] = new_solution[gene2]
    new_solution[gene2] = temp
    return new_solution
