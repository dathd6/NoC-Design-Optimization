import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

class Mesh2D:
    def __init__(self, n_cores, n_rows, n_cols, core_graph):
        self.n_cores = n_cores
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.core_graph = core_graph
        self.energy_consumption = None
        self.load_balancing = None
        self.random_core_mapping()
        self.find_shortest_route(is_random=True)
        self.calc_communication_cost()
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
        
    def random_core_mapping(self):
        # Mapping cores randomly on the routers by shuffling and reshaping
        self.mapping_order = list(range(self.n_cores)) + [-1] * (self.n_rows * self.n_cols - self.n_cores)
        np.random.shuffle(self.mapping_order)
        # Get core coordinate in router
        self.core_mapping_coord = {}
        for i in range(self.n_rows * self.n_cols):
            core = self.mapping_order[i]
            if core != -1:
                self.core_mapping_coord[core] = i

    def router_index_to_coordinates(self, idx):
        x = idx // self.n_cols
        y = idx % self.n_cols
        return (x, y)

    def find_shortest_route(self, is_random=True):
        self.routes = []
        for src, des, _ in self.core_graph:
            r1_x, r1_y = self.router_index_to_coordinates(self.core_mapping_coord[src])
            r2_x, r2_y= self.router_index_to_coordinates(self.core_mapping_coord[des])
            x_dir = self.find_shortest_route_direction(r1_x, r2_x)
            y_dir = self.find_shortest_route_direction(r1_y, r2_y)
            if is_random:
                dirs = [[0, 0] for _ in range(abs(r1_x - r2_x) + abs(r1_y - r2_y))]
                route = [0] * abs(r1_x - r2_x) + [1] * abs(r1_y - r2_y)
                np.random.shuffle(route)
                for i, step in enumerate(route):
                    if step == 0:
                        dirs[i][step] = x_dir 
                    elif step == 1:
                        dirs[i][step] = y_dir 
                self.routes.append(dirs)

    def calc_energy_consumption(self):
        # Calculate energy consumption
        pass

    def calc_communication_cost(self):
        self.com_cost = 0
        for i in range(len(self.core_graph)):
            _, _, bw = self.core_graph[i]
            n_hops = len(self.routes[i])
            self.com_cost += n_hops * bw

    def calc_average_load_degree(self):
        n_routers = self.n_rows * self.n_cols
        bw_throughput = np.zeros(n_routers)
        task_count = np.zeros(n_routers)
        total_bw = 0
        for i in range(len(self.core_graph)):
            src, _, bw = self.core_graph[i]
            total_bw += bw
            x, y = self.router_index_to_coordinates(self.core_mapping_coord[src])
            r = (x + 1) * (y + 1) - 1
            bw_throughput[r] = bw_throughput[r] + bw
            task_count[r] = task_count[r] + 1
            for x_dir, y_dir in self.routes[i]:
                x += x_dir
                y += y_dir
                r = (x + 1) * (y + 1) - 1
                bw_throughput[r] = bw_throughput[r] + bw
                task_count[r] = task_count[r] + 1

        self.avg_load_degree = 0
        for i in range(n_routers):
            if (task_count[i] == 0):
                continue
            self.avg_load_degree = self.avg_load_degree + bw_throughput[i] / (total_bw * task_count[i])
        self.avg_load_degree = self.avg_load_degree / n_routers

    def find_shortest_route_direction(self, x1, x2):
        if x1 < x2:
            return 1
        elif x1 > x2:
            return -1
        else:
            return 0
    
    def get_fitness(self):
        return [self.com_cost, self.avg_load_degree]

    def visualize_core_graph(self):
        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

       # Create a graph
        G = nx.Graph()

        # Add edges with bandwidth as weight
        for core1, core2, bandwidth in self.core_graph:
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
            core = f"C_{self.mapping_order[i]}"
            router = f"R_{i}"
            x, y = self.router_index_to_coordinates(i)
            nodes[core] = (x*2+ 1, y*2 + 1)
            nodes[router] = (x*2, y*2)

            # Add nodes to graph
            G.add_node(router, pos=nodes[router], label=router)

            # Connect IP to NI
            if self.mapping_order[i] != -1:
                G.add_node(core, pos=nodes[core], label=core)
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

        return G, nodes


    def visualize_NoC(self):
        G, nodes = self.create_noc_topology()
        pos = {node: nodes[node] for node in G.nodes()}
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        
        # Draw the network
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='skyblue', font_size=9, font_weight='bold', edge_color='gray', width=2)
        plt.title("Network on Chip (NoC) Topology")
        plt.axis('off')  # Turn off the axis
        plt.show()
