# multi_agent_rollout.py

import networkx as nx


def multi_agent_rollout(G: nx.DiGraph, agents: list, value_function: dict):
    """
    Perform deterministic multi-agent rollout based on Recursive Logit value function.

    Args:
        G (nx.DiGraph): Network graph.
        agents (list): List of agents.
        value_function (dict): Solved value function.

    Returns:
        agent_paths (list): List of agent path dictionaries.
        link_flows (dict): Dictionary {link_id: total flow assigned}.
    """
    agent_paths = []
    link_flows = {attr['link_id']: 0 for _, _, attr in G.edges(data=True)}

    for agent in agents:
        path_nodes = []
        path_links = []
        total_travel_time = 0.0
        total_free_flow_time = 0.0

        current_node = agent['origin_node']

        while current_node != agent['destination_node']:
            best_score = -float('inf')
            best_successor = None
            best_edge_attr = None

            for _, successor, attr in G.out_edges(current_node, data=True):
                score = -(attr['free_flow_travel_time'] + value_function[successor])
                if score > best_score:
                    best_score = score
                    best_successor = successor
                    best_edge_attr = attr

            if best_successor is None:
                print(f"Warning: Agent {agent['agent_id']} got stuck at node {current_node}")
                break

            # Record path
            path_nodes.append(current_node)
            path_links.append(best_edge_attr['link_id'])

            # Update totals
            total_travel_time += best_edge_attr['free_flow_travel_time']
            total_free_flow_time += best_edge_attr['free_flow_travel_time']

            # Update flow
            link_flows[best_edge_attr['link_id']] += 1

            # Move forward
            current_node = best_successor

        path_nodes.append(agent['destination_node'])

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
