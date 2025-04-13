import tensorflow as tf
import joblib
import numpy as np
import networkx as nx
import pandas as pd
import random
from tensorflow.keras.models import load_model

# Load scaler and LSTM model
scaler = joblib.load("main/scaler.pkl")
model = load_model("main/lstm_traffic_predictor_tf.keras")

# Load network topology from file
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
NUM_ANTS = 10
ITERATIONS = 25
EVAPORATION_RATE = 0.1
ALPHA, BETA, GAMMA = 1.0, 2.0, 1.5
Q = 100

pheromone = np.ones((NUM_NODES, NUM_NODES))
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

def aco_pathfinding(source, destination):
    """ ACO with real-time updates of network conditions and increasing timeframe """
    global pheromone
    best_path, best_cost = None, float('inf')

    for _ in range(ITERATIONS):
        update_network_conditions()  # Update parameters dynamically every iteration
        
        paths, path_costs = [], []

        for _ in range(NUM_ANTS):
            visited = [source]
            current_node = source

            while current_node != destination:
                neighbors = list(G.neighbors(current_node))
                next_node = random.choice(neighbors) if neighbors else None
                if next_node is None or next_node in visited:
                    break
                visited.append(next_node)
                current_node = next_node

            if visited[-1] == destination:
                total_latency = sum(G[visited[i]][visited[i+1]]['latency'] for i in range(len(visited) - 1))
                total_utilization = sum(G[visited[i]][visited[i+1]]['utilization'] for i in range(len(visited) - 1))
                path_cost = total_latency + (total_utilization * 0.5)
                paths.append(visited)
                path_costs.append(path_cost)

                if path_cost < best_cost:
                    best_path, best_cost = visited, path_cost

        pheromone *= (1 - EVAPORATION_RATE)  # Pheromone evaporation
        for path, cost in zip(paths, path_costs):
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i+1]] += Q / cost

    if best_path is None:
        raise Exception("No valid path found - Check dataset topology!")

    print(f"Best Path: {best_path}, Cost: {best_cost}")
    return best_path, best_cost

if __name__ == "__main__":
    source_node = 3
    destination_node = 10

    best_path, best_cost = aco_pathfinding(source_node, destination_node)
    print(f"ACO-LSTM Final Path: {best_path}, Cost: {best_cost}")