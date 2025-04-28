import networkx as nx
import numpy as np

def multi_agent_rollout(G: nx.DiGraph, agents: list, value_function_dict: dict, mu=0.1, random_seed=None):
    """
    Perform stochastic multi-agent rollout based on destination-specific value functions.

    Args:
        G (nx.DiGraph): Network graph.
        agents (list): List of agent dictionaries.
        value_function_dict (dict): {destination_zone: value_function dictionary}.
        mu (float): Softmax temperature parameter (smaller = more greedy).
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        agent_paths (list): List of agent path dictionaries.
        link_flows (dict): Dictionary {link_id: total flow assigned}.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    agent_paths = []
    link_flows = {attr['link_id']: 0 for _, _, attr in G.edges(data=True)}

    for agent in agents:
        origin = agent['origin_node']
        destination_zone = agent['destination_node']

        value_function = value_function_dict.get(destination_zone)
        if value_function is None:
            print(f"Warning: No value function for destination {destination_zone}. Skipping agent {agent['agent_id']}.")
            continue

        path_nodes = []
        path_links = []
        total_travel_time = 0.0
        total_free_flow_time = 0.0

        current_node = origin

        while current_node != destination_zone:
            successors = list(G.successors(current_node))

            if not successors:
                print(f"Warning: Agent {agent['agent_id']} stuck at node {current_node}. No successors.")
                break

            scores = []
            candidates = []

            for successor in successors:
                edge_data = G.get_edge_data(current_node, successor)
                link_cost = edge_data.get('current_travel_time', edge_data['free_flow_travel_time'])
                utility = -link_cost + value_function.get(successor, -np.inf)

                if np.isfinite(utility):
                    scores.append(utility / mu)
                    candidates.append((successor, edge_data))

            if not scores:
                print(f"Warning: No feasible moves for agent {agent['agent_id']} at node {current_node}.")
                break

            # Softmax probabilities
            exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
            probs = exp_scores / np.sum(exp_scores)

            # Randomly sample next move
            idx = np.random.choice(len(candidates), p=probs)
            next_node, next_edge_data = candidates[idx]

            # Record path
            path_nodes.append(current_node)
            path_links.append(next_edge_data['link_id'])

            total_travel_time += next_edge_data.get('current_travel_time', next_edge_data['free_flow_travel_time'])
            total_free_flow_time += next_edge_data['free_flow_travel_time']

            link_flows[next_edge_data['link_id']] += 1

            current_node = next_node

        path_nodes.append(destination_zone)

        agent_paths.append({
            'agent_id': agent['agent_id'],
            'o_zone_id': agent['origin_node'],
            'd_zone_id': agent['destination_node'],
            'path_length': len(path_links),
            'path_travel_time': total_travel_time,
            'path_free_flow_travel_time': total_free_flow_time,
            'link_sequence': path_links,
            'node_sequence': path_nodes
        })

    return agent_paths, link_flows
