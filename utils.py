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
