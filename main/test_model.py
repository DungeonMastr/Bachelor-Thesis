import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import random
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_rel
from ACO_PF import aco_pathfinding, G
from round_robin import weighted_round_robin_pathfinding
from tensorflow.keras.models import load_model

# Load trained LSTM model
LSTM_MODEL_PATH = "main/lstm_traffic_predictor_tf.keras"
scaler = joblib.load("main/scaler.pkl")
model = load_model(LSTM_MODEL_PATH)

# Load Traffic Data
df = pd.read_csv("simulated_network_traffic.csv")

# Parameters
NUM_TESTS = 20  # Number of paths to test

# Data Storage
aco_costs, wrr_costs = [], []
aco_times, wrr_times = [], []
aco_nodes_used, wrr_nodes_used = [], []
aco_utilization, wrr_utilization = [], []

# Randomly select test paths
nodes = list(G.nodes)
test_paths = [(random.choice(nodes), random.choice(nodes)) for _ in range(NUM_TESTS)]

# Run tests
for i, (source, destination) in enumerate(test_paths):
    print(f"\nRunning Test {i+1}/{NUM_TESTS} | Source: {source}, Destination: {destination}")

    # Reset graph before each test to ensure utilization tracking resets
    G.clear()
    for _, row in df.iterrows():
        src, dst = int(row["source_node"]), int(row["destination_node"])
        latency, bandwidth, utilization = row["end_to_end_latency_ms"], row["bandwidth_mbps"], row["utilization_percent"]
        G.add_edge(src, dst, latency=latency, bandwidth=bandwidth, utilization=utilization)

    # ACO Pathfinding
    start_time = time.time()
    best_path_aco, best_cost_aco = aco_pathfinding(source, destination)
    exec_time_aco = time.time() - start_time

    # WRR Pathfinding
    start_time = time.time()
    best_path_wrr, best_cost_wrr, avg_bandwidth_wrr = weighted_round_robin_pathfinding(source, destination)
    exec_time_wrr = time.time() - start_time

    # Ensure WRR path is valid before adding to stats
    if best_path_wrr is None:
        print(f"WRR Failed to find a path for {source} -> {destination}, Skipping this test.")
        continue

    # Store Results
    aco_costs.append(best_cost_aco)
    wrr_costs.append(best_cost_wrr)
    aco_times.append(exec_time_aco)
    wrr_times.append(exec_time_wrr)
    
    # Number of nodes used
    aco_nodes_used.append(len(set(best_path_aco)))
    wrr_nodes_used.append(len(set(best_path_wrr)))

    # **Fix: Collect Utilization Per Node**
    aco_util_vals = [G[u][v]['utilization'] for u, v in zip(best_path_aco[:-1], best_path_aco[1:])]
    wrr_util_vals = [G[u][v]['utilization'] for u, v in zip(best_path_wrr[:-1], best_path_wrr[1:])]

    aco_utilization.append(np.mean(aco_util_vals))
    wrr_utilization.append(np.mean(wrr_util_vals))

print("\n--- Statistical Analysis ---")

# Compute Statistics
def compute_stats(name, data_aco, data_wrr):
    mean_aco, mean_wrr = np.mean(data_aco), np.mean(data_wrr)
    sd_aco, sd_wrr = np.std(data_aco, ddof=1), np.std(data_wrr, ddof=1)
    t_stat, p_value = ttest_rel(data_aco, data_wrr)

    print(f"{name}:\n"
          f"  - ACO  | Mean: {mean_aco:.2f}, SD: {sd_aco:.4f}\n"
          f"  - WRR  | Mean: {mean_wrr:.2f}, SD: {sd_wrr:.4f}\n"
          f"  - t-test | t: {t_stat:.2f}, p: {p_value:.4f}\n")

compute_stats("Path Cost", aco_costs, wrr_costs)
compute_stats("Execution Time", aco_times, wrr_times)
compute_stats("Nodes Used", aco_nodes_used, wrr_nodes_used)
compute_stats("Utilization %", aco_utilization, wrr_utilization)

# **📊 Visualization**
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# **Plot 1: Path Cost Comparison**
labels = ["ACO", "WRR"]
axes[0, 0].boxplot([aco_costs, wrr_costs], labels=labels)
axes[0, 0].set_title("Path Cost Distribution")
axes[0, 0].set_ylabel("Cost (ms + Utilization)")

# **Plot 2: Execution Time Comparison**
axes[0, 1].boxplot([aco_times, wrr_times], labels=labels)
axes[0, 1].set_title("Execution Time Distribution")
axes[0, 1].set_ylabel("Time (s)")

# **Plot 3: Load Distribution - Nodes Used**
axes[0, 2].boxplot([aco_nodes_used, wrr_nodes_used], labels=labels)
axes[0, 2].set_title("Nodes Used in Paths")
axes[0, 2].set_ylabel("Number of Nodes")

# **Plot 4: Load Distribution - Avg Utilization %**
axes[1, 0].boxplot([aco_utilization, wrr_utilization], labels=labels)
axes[1, 0].set_title("Avg Utilization % on Path")
axes[1, 0].set_ylabel("Utilization %")

# Ensure `x` matches the number of recorded utilizations
x_aco = np.arange(len(aco_utilization))
x_wrr = np.arange(len(wrr_utilization))

# **Plot 5: Utilization Comparison by Path**
axes[1, 1].plot(x_aco, aco_utilization, label="ACO", marker="o", linestyle="--", color="blue")
axes[1, 1].plot(x_wrr, wrr_utilization, label="WRR", marker="s", linestyle="-", color="orange")
axes[1, 1].set_title("Utilization % Across Paths")
axes[1, 1].set_xlabel("Successful Test Paths")
axes[1, 1].set_ylabel("Utilization %")
axes[1, 1].legend()

# Hide empty subplot
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()


