import numpy as np
import networkx as nx
import pandas as pd
import random
from LSTM_predictor import predict_future_conditions  # Import LSTM function

# Load network topology from CSV
df_traffic = pd.read_csv("simulated_network_traffic.csv")

# Construct a graph from dataset (Static topology)
G = nx.Graph()
for _, row in df_traffic.iterrows():
    src, dst = int(row["source_node"]), int(row["destination_node"])
    latency, bandwidth, utilization = row["end_to_end_latency_ms"], row["bandwidth_mbps"], row["utilization_percent"]
    
    # Add edge with initial values
    G.add_edge(src, dst, latency=latency, bandwidth=bandwidth, utilization=utilization)

# Ensure the graph is fully connected
if not nx.is_connected(G):
    raise Exception("Graph is not fully connected! Ensure dataset consistency.")

# Constants for ACO (Optimized)
NUM_NODES = G.number_of_nodes()
NUM_ANTS = 5  # Reduced to improve speed
ITERATIONS = 5  # Lower iterations for debugging
EVAPORATION_RATE = 0.2  # Increased evaporation rate
ALPHA, BETA = 1.0, 2.0
Q = 50  # Lower pheromone deposit factor

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

    print(f"[DEBUG] Timeframe: {timeframe} | Network Conditions Updated")

def aco_pathfinding_lstm(source, destination):
    """ ACO with LSTM-enhanced network predictions """
    global pheromone
    best_path, best_cost = None, float('inf')

    print(f"[DEBUG] Starting ACO-LSTM Pathfinding from {source} -> {destination}")

    for iteration in range(ITERATIONS):
        print(f"[DEBUG] Iteration {iteration+1}/{ITERATIONS} - Updating Network Conditions...")
        update_network_conditions()  # Update network conditions dynamically every iteration
        
        paths, path_costs = [], []

        for ant in range(NUM_ANTS):
            visited = [source]
            current_node = source

            print(f"[DEBUG] Ant {ant+1}/{NUM_ANTS} - Starting at node {source}")

            while current_node != destination:
                neighbors = [n for n in G.neighbors(current_node) if n not in visited]
                
                if not neighbors:
                    print(f"[DEBUG] Ant {ant+1} - No neighbors found, breaking early.")
                    break  # Dead-end case

                # Batch predict for all neighbors
                batch_inputs = [
                    (G[current_node][neighbor]['latency'], G[current_node][neighbor]['bandwidth'], G[current_node][neighbor]['utilization'])
                    for neighbor in neighbors
                ]
                batch_predictions = [predict_future_conditions(*x) for x in batch_inputs]

                probabilities = []
                for i, neighbor in enumerate(neighbors):
                    if i >= len(batch_predictions):
                        continue  # Avoid IndexError

                    latency_change, bandwidth_change, utilization_change = batch_predictions[i]

                    # Get real-time values
                    real_latency = G[current_node][neighbor]['latency']
                    real_utilization = G[current_node][neighbor]['utilization']

                    # Apply batch-predicted values
                    adjusted_latency = real_latency + latency_change
                    adjusted_utilization = real_utilization + utilization_change

                    # Compute path cost using LSTM-adjusted values
                    cost = adjusted_latency + (adjusted_utilization * 0.5)

                    # Compute probability of selecting this edge
                    tau = pheromone[current_node][neighbor] ** ALPHA
                    eta = (1.0 / cost) ** BETA
                    probabilities.append((neighbor, tau * eta))

                if not probabilities:
                    print(f"[DEBUG] Ant {ant+1} - No valid moves left, stopping early.")
                    break  # No available paths

                # Select next node
                nodes, weights = zip(*probabilities)
                total = sum(weights)
                probabilities = [w / total for w in weights]

                next_node = random.choices(nodes, probabilities)[0]
                visited.append(next_node)
                current_node = next_node

                print(f"[DEBUG] Ant {ant+1} - Moved to node {next_node}")

            if visited[-1] == destination:
                total_latency = sum(G[visited[i]][visited[i+1]]['latency'] for i in range(len(visited) - 1))
                total_utilization = sum(G[visited[i]][visited[i+1]]['utilization'] for i in range(len(visited) - 1))
                path_cost = total_latency + (total_utilization * 0.5)
                
                paths.append(visited)
                path_costs.append(path_cost)

                if path_cost < best_cost:
                    best_path, best_cost = visited, path_cost

                print(f"[DEBUG] Ant {ant+1} - Found complete path: {visited} (Cost: {path_cost})")

        # Pheromone update (evaporation + reinforcement)
        pheromone *= (1 - EVAPORATION_RATE)
        for path, cost in zip(paths, path_costs):
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i+1]] += Q / cost  # Reinforce better paths

    if best_path is None:
        raise Exception("No valid path found - Check dataset topology!")

    print(f"[DEBUG] ACO-LSTM Final Path: {best_path}, Cost: {best_cost}")
    return best_path, best_cost

# Example Usage
if __name__ == "__main__":
    source_node = 3
    destination_node = 10

    best_path, best_cost = aco_pathfinding_lstm(source_node, destination_node)
    print(f"ACO Final Path: {best_path}, Cost: {best_cost}")




