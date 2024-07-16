# Import libraries and dependencies
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

def find_shortest_route_direction(x1, x2):
    if x1 < x2:
        return 1
    elif x1 > x2:
        return -1
    else:
        return 0

def router_index_to_coordinates(idx, n_cols):
    x = idx // n_cols
    y = idx % n_cols
    return (int(x), int(y))

def get_core_mapping_dict(core_mapping):
    core_mapping_dict = {}
    for i, c in enumerate(core_mapping):
        if c != -1:
            core_mapping_dict[c] = i
    return core_mapping_dict

def random_shortest_routing(core_graph, mapping_seq, n_cols, direction=False):
    # Random routing
    route_path = []

    mapping_dict = get_core_mapping_dict(mapping_seq)

    for i in range(len(core_graph)):
        src, des, _ = core_graph[i]
        # get x, y coordinate of each router in 2d mesh
        r1_x, r1_y = router_index_to_coordinates(mapping_dict[src], n_cols)
        r2_x, r2_y = router_index_to_coordinates(mapping_dict[des], n_cols)
        # Generate random route (number of feasible routes is |x1 - x2| * |y1 - y2|)
        if not direction:
            x_step = 0
            y_step = 1
        else:
            # Get the direction of x and y when finding the shortest route from router 1 to router 2
            x_dir = find_shortest_route_direction(r1_x, r2_x)
            y_dir = find_shortest_route_direction(r1_y, r2_y)
            x_step = x_dir * n_cols
            y_step = y_dir
        route = [x_step] * abs(r1_x - r2_x) + [y_step] * abs(r1_y - r2_y)
        np.random.shuffle(route)

        route_path.append(np.array(route))

    return route_path

def evaluation(self):
    f1 = calc_energy_consumption(
        mapping_seqs=self.mapping_seqs,
        n_cols=self.n_cols,
        core_graph=self.core_graph,
        es_bit=self.es_bit,
        el_bit=self.el_bit,
    ).reshape(-1, 1)
    f2 = calc_load_balance(
        n_rows=self.n_rows,
        n_cols=self.n_cols,
        mapping_seqs=self.mapping_seqs,
        route_paths=self.route_paths,
        core_graph=self.core_graph,
    ).reshape(-1, 1)

    self.f = np.concatenate((f1, f2), axis=1) 

def calc_energy_consumption(mapping_seqs, n_cols, core_graph, es_bit, el_bit):
    energy_consumption = np.zeros(len(mapping_seqs))
    for i in range(len(mapping_seqs)):
        m_dict = get_core_mapping_dict(mapping_seqs[i])
        for c1, c2, _ in core_graph:
            x1, y1 = router_index_to_coordinates(m_dict[c1], n_cols)
            x2, y2 = router_index_to_coordinates(m_dict[c2], n_cols)
            n_hops = np.abs(x1 - x2) + np.abs(y1 - y2)
            energy_consumption[i] = energy_consumption[i] + \
                (n_hops + 1) * es_bit + n_hops * el_bit
    return energy_consumption

def calc_energy_consumption_with_static_mapping_sequence(routing_paths, es_bit, el_bit):
    energy_consumption = np.zeros(len(routing_paths))

    for i in range(len(routing_paths)):
        for routing_path in routing_paths[i]:
            n_hops = len(routing_path)
            energy_consumption[i] = energy_consumption[i] + \
                (n_hops + 1) * es_bit + n_hops * el_bit

    return energy_consumption

def calc_load_balance_with_static_mapping_sequence(n_cols, n_rows, route_paths, mapping_seq, core_graph):
    size_p = len(route_paths)
    load_balance = np.zeros(size_p)
    n_routers = n_rows * n_cols

    seq = get_core_mapping_dict(mapping_seq)

    for k in range(size_p):
        bw_throughput = np.zeros(n_routers)   # Bandwidth throughput in each router
        task_count = np.zeros(n_routers)   # Bandwidth throughput in each router

        route_path = route_paths[k]
        
        for i in range(len(core_graph)):
            src, _, bw = core_graph[i]

            tc = np.zeros(n_routers)
            r_src = int(seq[src])
            tc[r_src] = 1
            for step in route_path[i]:
                r_src = r_src + step
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

def calc_load_balance(n_cols, n_rows, route_paths, mapping_seqs, core_graph):
    size_p = len(route_paths)
    load_balance = np.zeros(size_p)
    n_routers = n_rows * n_cols


    for k in range(size_p):
        bw_throughput = np.zeros(n_routers)   # Bandwidth throughput in each router
        task_count = np.zeros(n_routers)   # Bandwidth throughput in each router

        route_path = route_paths[k]
        m_dict = get_core_mapping_dict(mapping_seqs[k])

        for i in range(len(core_graph)):
            src, des, bw = core_graph[i]
            # get x, y coordinate of each router in 2d mesh
            r1_x, r1_y = router_index_to_coordinates(m_dict[src], n_cols)
            r2_x, r2_y = router_index_to_coordinates(m_dict[des], n_cols)
            # Get the direction of x and y when finding the shortest route from router 1 to router 2
            x_dir = find_shortest_route_direction(r1_x, r2_x)
            y_dir = find_shortest_route_direction(r1_y, r2_y)

            tc = np.zeros(n_routers)
            r_src = int(m_dict[src])
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
