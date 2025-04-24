import matplotlib.pyplot as plt
import networkx as nx
from network_classes import Node, Link, Network
from demand_classes import DemandSet
from utility_func import UtilityFunction
from recursive_logit import RecursiveLogitModel
from rl_static_assignment import RLStaticAssigner
from multiagent_rollout import MultiAgentPolicyIteration, VehicleAgent

def main():
    # 1) Build a small DAG network in code (or parse from CSV)
    net = Network()
    # Create some nodes
    for i in range(1, 6):
        net.add_node(Node(node_id=i))

    # Suppose link_id=some integer or string, from->to
    # Let's build a small DAG:
    net.add_link(Link(link_id='L1', start_node=1, end_node=2,
                      attributes={'length':2.0, 'travel_time':1.0}))
    net.add_link(Link(link_id='L2', start_node=2, end_node=3,
                      attributes={'length':3.0, 'travel_time':2.0}))
    net.add_link(Link(link_id='L3', start_node=2, end_node=4,
                      attributes={'length':1.0, 'travel_time':0.5}))
    net.add_link(Link(link_id='L4', start_node=4, end_node=5,
                      attributes={'length':2.0, 'travel_time':2.0}))
    net.add_link(Link(link_id='L5', start_node=3, end_node=5,
                      attributes={'length':1.0, 'travel_time':1.0}))

    # We treat node 5 as a pseudo-destination that might have a dummy link
    # but for simplicity let's say link_id='DestLink' goes from 5->5. (Absorbing.)
    # Or we can just treat link 5 as absorbing if it has no successors.
    # If you prefer a special dummy link, do that:
    net.add_link(Link(link_id='D', start_node=5, end_node=5,
                      attributes={'length':0.0, 'travel_time':0.0}))

    # We don't strictly need topological order at the node level for the RL model,
    # but let's do it if we want it for something else:
    net.build_topological_order()

    # convert and draw the graph
    G = net.to_networkx()

    plt.figure(figsize=(10, 6))
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=600, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['id'] for u, v, d in G.edges(data=True)})
    plt.title("Network Structure")
    plt.show()

    # 2) Build a demand set
    demands = DemandSet()
    # Suppose we say origin is L1 (which starts at node 1->2) for an OD volume of 100
    # and the "destination link" is 'D' (the absorbing link).
    # In many designs, you'd pick a special link from node=1->1 as the origin, but let's keep it simple:
    demands.add_demand('L1', 'D', 100.0)

    # 3) Build the utility function & RL model
    beta_params = {'time': -0.5, 'distance': -0.2}  # negative means we prefer less time/distance
    util_func = UtilityFunction(beta_params)
    rl_model = RecursiveLogitModel(net, util_func, mu=1.0)

    # 4) Use MultiAgentRollout
    vehicles = []
    for record in demands.demands:
        for i in range(int(record.volume)):
            agent_id = f"{record.origin_link_id}_{record.destination_link_id}_{i}"
            vehicles.append(VehicleAgent(agent_id, record.origin_link_id, record.destination_link_id))

    pi = MultiAgentPolicyIteration(net, rl_model, vehicles)
    pi.policy_iteration(n_iters=100, patience=10)
    flows = pi.get_link_flows()

    # 5) Print results
    print("\nLink Flows from Final Policy after Policy Iteration:")
    for link_id, flow_val in sorted(flows.items()):
        print(f"  Link {link_id}: Flow = {flow_val:.1f}")

    # Plot reward convergence curve
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