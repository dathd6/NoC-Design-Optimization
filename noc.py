# Import libraries and dependencies
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from utils import find_shortest_route_direction, get_task_throughput

class NetworkOnChip:
    # Mesh 2D topology
    def __init__(self, n_cores, n_rows, n_cols, es_bit, el_bit, core_graph, mapping_seq=None, routes=None, bw_throughput=None, task_count=None):
        """
        Mesh 2D Attributes
            n_cores         - number of cores
            n_rows & n_cols - number of rows and columns
            core_graph      - core graph/task graph
            es_bit & el_bit - energy consumed in the switchs (routers) and links
        """
        self.n_cores = n_cores
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.core_graph = core_graph # ~ Task graph
        self.es_bit = es_bit
        self.el_bit = el_bit

        # Fitnesses - Objective values
        self.energy_consumption = None
        self.avg_load_degree = None

        # Decision variables
        if mapping_seq == None:
            self.mapping_seq = np.ones(n_rows * n_cols) * -1 # Core mapping sequence for each router
            self.core_mapping_coord = {}                     # Dictionary with key/value is core/router
            self.random_core_mapping()
        else:
            self.mapping_seq = mapping_seq
            self.get_core_mapping_coordinate()

        if routes == None and bw_throughput == None and task_count == None:
            self.routes = []                                 # The data transfer route for each task in core graph
            self.bw_throughput = np.zeros(n_rows * n_cols)   # Bandwidth throughput in each router
            self.task_count = np.zeros(n_rows * n_cols)      # Number of task go through each router
            self.random_shortest_routing()
        else:
            self.routes = routes
            self.bw_throughput = bw_throughput
            self.task_count = task_count

        # Calculate fitnesses
        self.calc_energy_consumption()
        self.calc_average_load_degree()

    def __gt__(self, other):
        # Dominance
        s_fitness = self.get_fitness()
        o_fitness = other.get_fitness()
        result = True
        for i in range(len(s_fitness)):
            result = result and (s_fitness[i] < o_fitness[i])
        return result

    def __eq__(self, other):
        # Equal
        s_fitness = self.get_fitness()
        o_fitness = other.get_fitness()
        result = True
        for i in range(len(s_fitness)):
            result = result and (s_fitness[i] == o_fitness[i])
        return result

    def __ge__(self, other):
        # Weakly dominance
        s_fitness = self.get_fitness()
        o_fitness = other.get_fitness()
        result = True
        for i in range(len(s_fitness)):
            result = result and (s_fitness[i] <= o_fitness[i])
        return result
    
    def get_core_mapping_coordinate(self):
        self.core_mapping_coord = {}
        for i in range(self.n_rows * self.n_cols):
            core = self.mapping_seq[i]
            if core != -1:
                self.core_mapping_coord[core] = i
        
    def random_core_mapping(self):
        # Mapping cores randomly on the routers by shuffling
        self.mapping_seq = list(range(self.n_cores)) + [-1] * (self.n_rows * self.n_cols - self.n_cores)
        np.random.shuffle(self.mapping_seq)
        self.get_core_mapping_coordinate()

    def random_shortest_routing(self):
        # Random routing
        self.routes = []
        for src, des, bw in self.core_graph:
            route, bw_throughput, task_count = self.find_shortest_route(
                self.core_mapping_coord[src],
                self.core_mapping_coord[des],
                bw
            )
            self.routes.append(route)
            self.bw_throughput = self.bw_throughput + bw_throughput
            self.task_count = self.task_count + task_count

    def router_index_to_coordinates(self, idx):
        x = idx // self.n_cols
        y = idx % self.n_cols
        return (x, y)

    def find_shortest_route(self, router_1, router_2, bw):
        # Get x, y coordinate of each router in 2D mesh
        r1_x, r1_y = self.router_index_to_coordinates(router_1)
        r2_x, r2_y = self.router_index_to_coordinates(router_2)
        # Get the direction of x and y when finding the shortest route from router 1 to router 2
        x_dir = find_shortest_route_direction(r1_x, r2_x)
        y_dir = find_shortest_route_direction(r1_y, r2_y)

        # Generate random route (number of feasible routes is |x1 - x2| * |y1 - y2|)
        route = [0] * abs(r1_x - r2_x) + [1] * abs(r1_y - r2_y)
        np.random.shuffle(route)

        # Get bandwidth throughput and task count through every router (switch)
        # that the data transfer through
        bw_throughput, task_count = get_task_throughput(
            n_routers=self.n_rows * self.n_cols,
            start=(r1_x, r1_y),
            n_cols=self.n_cols,
            dir=(x_dir, y_dir),
            route=route,
            bw=bw,
        )

        return route, bw_throughput, task_count

    def calc_energy_consumption(self):
        # Search all the routes to calculate total energy consumption
        self.energy_consumption = 0
        for route in self.routes:
            n_hops = len(route)
            self.energy_consumption = self.energy_consumption + \
                (n_hops + 1) * self.es_bit + n_hops * self.el_bit

    # Not use fitness - Communication cost
    def calc_communication_cost(self):
        self.com_cost = 0
        for i in range(len(self.core_graph)):
            _, _, bw = self.core_graph[i]
            n_hops = len(self.routes[i])
            self.com_cost += n_hops * bw

    def calc_average_load_degree(self):
        n_routers = self.n_rows * self.n_cols

        # Calculate the total bandwidth
        total_bw = 0
        for _, _, bw in self.core_graph:
            total_bw += bw

        self.avg_load_degree = 0
        # Search all router that has task count
        # and calculate the load degree at that router
        # --> calculate the total load degree --> average load degree
        for i in range(n_routers):
            if (self.task_count[i] == 0):
                continue
            self.avg_load_degree = self.avg_load_degree + self.bw_throughput[i] / (total_bw * self.task_count[i])
        self.avg_load_degree = self.avg_load_degree / n_routers

    def get_fitness(self):
        return np.array([self.energy_consumption, self.avg_load_degree])

    def visualize_core_graph(self):
        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

       # Create a graph
        G = nx.Graph()

        # Add edges with bandwidth as weight
        for core1, core2, _ in self.core_graph:
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

    def create_noc_topology(self):
        G = nx.DiGraph()  # Directed graph to show possible directions of data flow

        # Creating nodes for IP, NI, R
        nodes = {}
        for i in range(self.n_rows * self.n_cols):
            core = f"C_{self.mapping_seq[i]}"
            router = f"R_{i}"
            x, y = self.router_index_to_coordinates(i)
            nodes[core] = (x*2+ 1, y*2 + 1)
            nodes[router] = (x*2, y*2)

            # Add nodes to graph
            G.add_node(router, pos=nodes[router], label=router, color='orange')

            # Connect IP to NI
            if self.mapping_seq[i] != -1:
                G.add_node(core, pos=nodes[core], label=core, color='lightblue')
                G.add_edge(core, router)

        # Creating edges between routers
        for i in range(self.n_rows * self.n_cols):
            x, y = self.router_index_to_coordinates(i)
            if x < self.n_rows - 1:  # Connect down
                G.add_edge(f"R_{x * self.n_cols + y}", f"R_{(x + 1) * self.n_cols + y}")
                G.add_edge(f"R_{(x + 1) * self.n_cols + y}", f"R_{x * self.n_cols + y}")
            if y < self.n_cols - 1:  # Connect right
                G.add_edge(f"R_{x * self.n_cols + y}", f"R_{x * self.n_cols + (y + 1)}")
                G.add_edge(f"R_{x * self.n_cols + (y + 1)}", f"R_{x * self.n_cols + y}")

        G.add_edge("R_0", "R_3", color='green')
        G.add_edge("R_0", "R_1", color='blue')
        G.add_edge("R_0", "R_4", color='yellow')

        return G, nodes


    def visualize_NoC(self):
        G, nodes = self.create_noc_topology()
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
