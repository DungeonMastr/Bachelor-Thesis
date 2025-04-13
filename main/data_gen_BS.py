import simpy
import random
import pandas as pd
import numpy as np
import networkx as nx

# Simulation Parameters
SIM_TIME = 1000  # Total simulation time in seconds
NUM_NODES = 300  # Total number of nodes in the network
NUM_SELECTED_NODES = 4  # Select 4 nodes for paths
PACKET_GEN_INTERVAL = 5  # Packet generation interval

# Randomly select 4 nodes for the subgraph (simulating a real network with limited paths)
selected_nodes = random.sample(range(NUM_NODES), NUM_SELECTED_NODES)
print(f"Selected nodes for path simulation: {selected_nodes}")

# Create a random graph with partial connectivity (no complete graph)
# The probability of creating an edge between any two nodes
p_edge = 0.1  # Adjust probability to control the number of edges
G = nx.erdos_renyi_graph(NUM_NODES, p_edge)

# Ensure that the graph is connected (if it is not, make it connected)
while not nx.is_connected(G):
    G = nx.erdos_renyi_graph(NUM_NODES, p_edge)

# Add random weights (latency and bandwidth) to edges
for u, v in G.edges():
    G[u][v]['latency'] = random.uniform(10, 100)  # Example latency (ms)
    G[u][v]['bandwidth'] = random.uniform(50, 500)  # Example bandwidth (Mbps)

# Data Storage
traffic_data = []
OUTPUT_FILE = "simulated_black_swan_traffic.csv"

class NetworkNode:
    def __init__(self, env, node_id, nodes):
        self.env = env
        self.node_id = node_id
        self.latency = random.uniform(10, 100)  # Initial latency (ms)
        self.bandwidth = random.uniform(50, 500)  # Initial bandwidth (Mbps)
        self.utilization = random.uniform(20, 80)  # Initial utilization (%)
        self.nodes = nodes  # Reference to all nodes

        # Start traffic generation process for this node
        self.action = env.process(self.generate_traffic())

    def generate_traffic(self):
        """Simulate network traffic with packet transmission and Black Swan events."""
        while True:
            # Apply traffic generation for selected nodes
            if self.node_id in selected_nodes:
                # Select a random destination node (only among selected nodes)
                destination = random.choice([node for node in selected_nodes if node != self.node_id])
                
                # Ensure the destination node is within the selected nodes list
                if destination in self.nodes:
                    # Calculate end-to-end latency (self latency + destination latency)
                    total_latency = self.latency + self.nodes[destination].latency

                    # Simulate normal traffic fluctuations
                    self.latency += np.random.normal(0, 5)
                    self.bandwidth += np.random.normal(0, 20)
                    self.utilization += np.random.normal(0, 10)

                    # Constrain values to realistic limits
                    self.latency = max(5, min(self.latency, 300))
                    self.bandwidth = max(10, min(self.bandwidth, 1000))
                    self.utilization = max(0, min(self.utilization, 100))

                    # Store data for the selected path between source and destination nodes
                    traffic_data.append({
                        "timestamp": self.env.now,
                        "source_node": self.node_id,
                        "destination_node": destination,
                        "latency_ms": round(self.latency, 2),
                        "end_to_end_latency_ms": round(total_latency, 2),
                        "bandwidth_mbps": round(self.bandwidth, 2),
                        "utilization_percent": round(self.utilization, 2),
                        "black_swan_event": 0  # No black swan event for simplicity
                    })

            # Wait until the next packet is generated
            yield self.env.timeout(random.expovariate(1.0 / PACKET_GEN_INTERVAL))

# Run Simulation
env = simpy.Environment()

# Initialize only the selected nodes for the simulation
nodes = [NetworkNode(env, node_id, []) for node_id in selected_nodes]

# Assign references so each node can access others in the selected subgraph
for node in nodes:
    node.nodes = nodes  # Each node now knows about the others

env.run(until=SIM_TIME)

# Convert to DataFrame and Overwrite the File
df = pd.DataFrame(traffic_data)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Simulation complete. Data saved as '{OUTPUT_FILE}'.")


