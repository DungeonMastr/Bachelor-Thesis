import random
import networkx as nx
import pandas as pd
import numpy as np

# Load network topology from the file
df_traffic = pd.read_csv("simulated_network_traffic.csv")

# Construct graph from dataset (Static topology)
G = nx.Graph()
for _, row in df_traffic.iterrows():
    src, dst = int(row["source_node"]), int(row["destination_node"])
    latency, bandwidth, utilization = row["end_to_end_latency_ms"], row["bandwidth_mbps"], row["utilization_percent"]
    G.add_edge(src, dst, latency=latency, bandwidth=bandwidth, utilization=utilization)

if not nx.is_connected(G):
    raise Exception("Graph is not fully connected!")

NUM_NODES = G.number_of_nodes()
timeframe = 0  # Start timeframe

def update_network_conditions():
    """ Dynamically update edge parameters while keeping topology static """
    global timeframe
    timeframe += 1  # Linear timeframe increment

    for u, v in G.edges():
        G[u][v]['latency'] += np.random.uniform(-2, 2)
        G[u][v]['bandwidth'] += np.random.uniform(-10, 10)
        G[u][v]['utilization'] += np.random.uniform(-3, 3)

        # Keep values within realistic bounds
        G[u][v]['latency'] = np.clip(G[u][v]['latency'], 10, 100)
        G[u][v]['bandwidth'] = np.clip(G[u][v]['bandwidth'], 50, 1000)
        G[u][v]['utilization'] = np.clip(G[u][v]['utilization'], 0, 100)

    print(f"Timeframe: {timeframe} | Network Conditions Updated")

def weighted_round_robin_pathfinding(source, destination):
    """ Weighted Round Robin Pathfinding Algorithm with real-time parameter updates """
    update_network_conditions()  # Dynamically update network conditions

    visited = {source}
    path = [source]
    total_latency = 0
    total_utilization = 0
    total_bandwidth = 0

    while path[-1] != destination:
        current_node = path[-1]
        neighbors = [n for n in G.neighbors(current_node) if n not in visited]

        if not neighbors:
            return None, float('inf'), 0  # No path found

        # Select next node based on lowest latency and highest bandwidth
        next_node = min(neighbors, key=lambda n: (G[current_node][n]['latency'], -G[current_node][n]['bandwidth']))
        path.append(next_node)
        visited.add(next_node)

        # Update path costs
        total_latency += G[current_node][next_node]['latency']
        total_utilization += G[current_node][next_node]['utilization']
        total_bandwidth += G[current_node][next_node]['bandwidth']

    path_cost = total_latency + (total_utilization * 0.5)
    print(f"Weighted Round Robin Path: {path}, Cost: {path_cost}, Avg Bandwidth: {total_bandwidth / len(path)}")
    return path, path_cost, total_bandwidth / len(path)





