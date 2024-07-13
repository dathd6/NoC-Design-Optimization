# Import libraries and dependencies
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from utils import find_shortest_route_direction, router_index_to_coordinates

def get_core_mappings(router_mappings, n_routers):
    core_mappings = []
    core_mapping = -np.ones(n_routers)
    for i, r in enumerate(router_mappings):
        core_mapping[r] = i
    core_mappings.append(core_mapping)
    return np.array(core_mappings)

def get_router_mappings(core_mappings, n_cores):
    router_mappings = []
    for core_mapping in core_mappings:
        router_mapping = np.zeros(n_cores)
        for i, c in enumerate(core_mapping):
            if c != -1:
                router_mapping[c] = i
        router_mappings.append(router_mapping)
    return np.array(router_mappings)

def calc_energy_consumption(mapping_seqs, n_cols, core_graph, es_bit, el_bit):
    n_cores = len(core_graph)
    router_mappings = get_router_mappings(mapping_seqs, n_cores)
    energy_consumption = np.zeros(len(mapping_seqs))
    for i in range(len(router_mappings)):
        router_mapping = router_mappings[i]
        for c1, c2, _ in core_graph:
            x1, y1 = router_index_to_coordinates(router_mapping[c1], n_cols)
            x2, y2 = router_index_to_coordinates(router_mapping[c2], n_cols)
            n_hops = np.abs(x1 - x2) + np.abs(y1 - y2)
            energy_consumption[i] = energy_consumption[i] + \
                (n_hops + 1) * es_bit + n_hops * el_bit
    return energy_consumption

def calc_load_balance(n_cols, n_rows, route_paths, mapping_seqs, core_graph):
    size_p = len(route_paths)
    load_balance = np.zeros(size_p)
    n_cores = len(core_graph)
    n_routers = n_rows * n_cols

    r_mapping_seqs = get_router_mappings(mapping_seqs, n_cores)

    for k in range(size_p):
        bw_throughput = np.zeros(n_routers)   # Bandwidth throughput in each router
        task_count = np.zeros(n_routers)   # Bandwidth throughput in each router

        route_path = route_paths[k]
        r_mapping_seq = r_mapping_seqs[k]

        for i in range(n_cores):
            src, des, bw = core_graph[i]
            # get x, y coordinate of each router in 2d mesh
            r1_x, r1_y = router_index_to_coordinates(r_mapping_seq[src], n_cols)
            r2_x, r2_y = router_index_to_coordinates(r_mapping_seq[des], n_cols)
            # Get the direction of x and y when finding the shortest route from router 1 to router 2
            x_dir = find_shortest_route_direction(r1_x, r2_x)
            y_dir = find_shortest_route_direction(r1_y, r2_y)

            tc = np.zeros(n_routers)
            r_src = int(r_mapping_seq[src])
            tc[r_src] = 1
            for step in route_path[i]:
                if step == 0:
                    r_src = r_src + x_dir * n_cols
                if step == 1:
                    r_src = r_src + y_dir
                tc[r_src] = 1

            bw_throughput = bw_throughput + tc * bw
            task_count = task_count + tc
            
        # Calculate the total bandwidth
        total_bw = 0
        for _, _, bw in core_graph:
            total_bw += bw

        load_degree = []
        for bw in bw_throughput:
            load_degree.append(bw / total_bw)

        avg_load_degree = 0
        # Search all router that has task count
        # and calculate the load degree at that router
        # --> calculate the total load degree --> average load degree
        for i in range(n_routers):
            if (task_count[i] == 0):
                continue
            avg_load_degree = avg_load_degree + load_degree[i] / task_count[i]
        avg_load_degree = avg_load_degree / n_routers

        lb = 0
        for i in range(n_routers):
            lb = lb + np.abs(load_degree[i] - avg_load_degree)

        load_balance[k] = lb

    return load_balance

def random_core_mapping(n_cores, n_rows, n_cols):
    # Mapping cores randomly on the routers by shuffling
    mapping_seq = list(range(n_cores)) + [-1] * (n_rows * n_cols - n_cores)
    np.random.shuffle(mapping_seq)

    return mapping_seq

def random_shortest_routing(core_graph, mapping_seq, n_cols):
    # Random routing
    n_cores = len(core_graph)
    route_path = []

    seq = get_router_mappings([mapping_seq], n_cores)[0]

    for i in range(n_cores):
        src, des, _ = core_graph[i]
        # get x, y coordinate of each router in 2d mesh
        r1_x, r1_y = router_index_to_coordinates(seq[src], n_cols)
        r2_x, r2_y = router_index_to_coordinates(seq[des], n_cols)
        # Generate random route (number of feasible routes is |x1 - x2| * |y1 - y2|)
        route = [0] * abs(r1_x - r2_x) + [1] * abs(r1_y - r2_y)
        np.random.shuffle(route)

        route_path.append(np.array(route))

    return route_path

def visualize_core_graph(core_graph):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

   # Create a graph
    G = nx.Graph()

    # Add edges with bandwidth as weight
    for core1, core2, _ in core_graph:
        G.add_edge(core1, core2)

    # Generate positions for each node
    pos = nx.spring_layout(G)  # Uses a force-directed layout

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)

    # Draw the edges with bandwidths as labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw the node labels
    nx.draw_networkx_labels(G, pos)

    # Remove axes
    plt.axis('off')

    # Show the plot
    plt.show()

def create_noc_topology(n_rows, n_cols, mapping_seq):
    G = nx.DiGraph()  # Directed graph to show possible directions of data flow

    # Creating nodes for IP, NI, R
    nodes = {}
    for i in range(n_rows * n_cols):
        core = f"C_{mapping_seq[i]}"
        router = f"R_{i}"
        x, y = router_index_to_coordinates(i, n_cols)
        nodes[core] = (x*2+ 1, y*2 + 1)
        nodes[router] = (x*2, y*2)

        # Add nodes to graph
        G.add_node(router, pos=nodes[router], label=router, color='orange')

        # Connect IP to NI
        if mapping_seq[i] != -1:
            G.add_node(core, pos=nodes[core], label=core, color='lightblue')
            G.add_edge(core, router)

    # Creating edges between routers
    for i in range(n_rows * n_cols):
        x, y = router_index_to_coordinates(i, n_cols)
        if x < n_rows - 1:  # Connect down
            G.add_edge(f"R_{x * n_cols + y}", f"R_{(x + 1) * n_cols + y}")
            G.add_edge(f"R_{(x + 1) * n_cols + y}", f"R_{x * n_cols + y}")
        if y < n_cols - 1:  # Connect right
            G.add_edge(f"R_{x * n_cols + y}", f"R_{x * n_cols + (y + 1)}")
            G.add_edge(f"R_{x * n_cols + (y + 1)}", f"R_{x * n_cols + y}")

    return G, nodes


def visualize_NoC(n_rows, n_cols, mapping_seq):
    G, nodes = create_noc_topology(n_rows, n_cols, mapping_seq)
    pos = {node: (loc[1], -loc[0]) for node, loc in nodes.items()}
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}

    node_color = [G.nodes[node]['color'] for node in G]
    edge_color = [G[u][v].get('color', 'gray') for u, v in G.edges()]
    
    # Draw the network
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_color, node_size=2000, font_size=9, font_weight='bold', edge_color=edge_color, width=2, arrows=True)
    plt.title("Network on Chip (NoC) Topology")
    plt.axis('off')  # Turn off the axis
    plt.show()
