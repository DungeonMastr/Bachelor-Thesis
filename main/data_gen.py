import random
import pandas as pd
import numpy as np

# Simulation Parameters
SIM_TIME = 250  # Total simulation steps
NUM_NODES = 100   # Number of network nodes
CONNECTIONS_PER_NODE = 4  # Fixed connections per node

# Initialize static network topology
network_topology = {node: random.sample([n for n in range(NUM_NODES) if n != node], CONNECTIONS_PER_NODE) for node in range(NUM_NODES)}

# Data Storage
traffic_data = []
OUTPUT_FILE = "LSTM_traffic.csv"

# Generate dataset
for t in range(SIM_TIME):
    for node in range(NUM_NODES):
        latency = np.clip(random.normalvariate(50, 10), 10, 100)
        bandwidth = np.clip(random.normalvariate(300, 50), 50, 1000)
        utilization = np.clip(random.normalvariate(50, 15), 0, 100)
        
        for dest_node in network_topology[node]:
            dest_latency = np.clip(random.normalvariate(50, 10), 10, 100)
            end_to_end_latency = latency + dest_latency

            traffic_data.append({
                "timestamp": t,
                "source_node": node,
                "destination_node": dest_node,
                "latency_ms": round(latency, 2),
                "end_to_end_latency_ms": round(end_to_end_latency, 2),
                "bandwidth_mbps": round(bandwidth, 2),
                "utilization_percent": round(utilization, 2)
            })

# Convert to DataFrame and save
pd.DataFrame(traffic_data).to_csv(OUTPUT_FILE, index=False)
print(f"Dataset generated: '{OUTPUT_FILE}'")
