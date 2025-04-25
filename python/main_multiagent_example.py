import matplotlib.pyplot as plt
import networkx as nx
import csv
import os
from network_classes import Node, Link, Network
from demand_classes import DemandSet
from utility_func import UtilityFunction
from recursive_logit import RecursiveLogitModel
# from rl_static_assignment import RLStaticAssigner
# from multiagent_rollout import MultiAgentPolicyIteration, VehicleAgent
from multiagent_rollout_test import MultiAgentPolicyIteration, VehicleAgent
from network_loader import load_network_from_csv, load_od_demand

def main():
    # 1) Load network and demand from CSV

    data_path = "../data_sets/3-corridor"
    node_file = os.path.join(data_path, "node.csv")
    link_file = os.path.join(data_path, "link.csv")
    demand_file = os.path.join(data_path, "demand.csv")

    net = load_network_from_csv(node_file, link_file)
    demands = load_od_demand(demand_file, net)

    # Optional: build topological order (only used in DAGs, safe to skip in cycles)
    net.build_topological_order()

    # 2) Visualize network
    G = net.to_networkx()
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=600, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['id'] for u, v, d in G.edges(data=True)})
    plt.title("Network Structure")
    plt.show()

    # 3) Build utility function and RL model
    beta_params = {'time': -0.5, 'distance': -0.2}  # Negative = prefer shorter time/distance
    util_func = UtilityFunction(beta_params)
    rl_model = RecursiveLogitModel(net, util_func, mu=1.0)

    # 4) Create agents for rollout
    vehicles = []
    for record in demands.demands:
        for i in range(int(record.volume)):
            agent_id = f"{record.origin_link_id}_{record.destination_link_id}_{i}"
            vehicles.append(VehicleAgent(agent_id,
                                         record.origin_link_id,
                                         record.destination_link_id,
                                         record.origin_zone,
                                         record.destination_zone
                                         ))

    # 5) Run policy iteration
    pi = MultiAgentPolicyIteration(net, rl_model, vehicles)
    pi.policy_iteration(n_iters=100, patience=10)
    flows = pi.get_link_flows()

    # Export link performance results
    with open(os.path.join(data_path, "link_performance.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["link_id", "from_node_id", "to_node_id", "flow", "travel_time", "free_flow_travel_time"])
        for link_id, link in net.links.items():
            flow = flows.get(link_id, 0.0)
            travel_time = link.attributes.get("travel_time", 0.0)
            free_flow_time = link.attributes.get("travel_time", 0.0)  # assumed equal here
            writer.writerow([link_id, link.start_node, link.end_node, flow, travel_time, free_flow_time])

    # Export agent results
    with open(os.path.join(data_path, "agent_result.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent_id", "o_zone_id", "d_zone_id", "path_length",
            "path_travel_time", "path_free_flow_travel_time",
            "link_sequence", "node_sequence"
        ])
        for agent in vehicles:
            writer.writerow([
                agent.agent_id,
                agent.o_zone, agent.d_zone,
                len(agent.path),
                round(agent.path_travel_time, 3),
                round(agent.path_free_flow_time, 3),
                "->".join(agent.path),
                "->".join(map(str, agent.path_nodes))
            ])

    # 6) Print results
    print("\nLink Flows from Final Policy after Policy Iteration:")
    for link_id, flow_val in sorted(flows.items()):
        print(f"  Link {link_id}: Flow = {flow_val:.1f}")

    # 7) Plot convergence
    plt.figure(figsize=(8, 6))
    plt.plot(pi.reward_history, marker='o', linestyle='-')
    plt.title("Average Reward over Policy Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward (−Travel Time)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
