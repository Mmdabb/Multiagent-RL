# multi_agent_rollout.py

import networkx as nx
import numpy as np

def multi_agent_rollout(G: nx.DiGraph, agents: list, value_function_dict: dict):
    """
    Perform deterministic multi-agent rollout based on destination-specific value functions.

    Args:
        G (nx.DiGraph): Network graph.
        agents (list): List of agents.
        value_function_dict (dict): {destination_zone: value_function dictionary}.

    Returns:
        agent_paths (list): List of agent path dictionaries.
        link_flows (dict): Dictionary {link_id: total flow assigned}.
    """
    agent_paths = []
    link_flows = {attr['link_id']: 0 for _, _, attr in G.edges(data=True)}

    for agent in agents:
        origin = agent['origin_node']
        destination_zone = agent['destination_node']

        # Get the value function for the agent's destination
        value_function = value_function_dict.get(destination_zone)
        if value_function is None:
            print(f"Warning: No value function found for destination {destination_zone}. Agent {agent['agent_id']} skipped.")
            continue

        path_nodes = []
        path_links = []
        total_travel_time = 0.0
        total_free_flow_time = 0.0

        current_node = origin

        while current_node not in value_function or value_function[current_node] == -np.inf:
            print(f"Warning: Agent {agent['agent_id']} starts from node {current_node} with no path to destination {destination_zone}.")
            break

        while current_node != destination_zone:
            best_score = -float('inf')
            best_successor = None
            best_edge_attr = None

            for _, successor, attr in G.out_edges(current_node, data=True):
                # link_cost = attr['free_flow_travel_time']
                link_cost = attr.get('current_travel_time', attr['free_flow_travel_time'])

                utility = -link_cost
                score = utility + value_function.get(successor, -np.inf)

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
            total_travel_time += best_edge_attr['current_travel_time']
            total_free_flow_time += best_edge_attr['free_flow_travel_time']

            # Update flow
            link_flows[best_edge_attr['link_id']] += 1

            # Move forward
            current_node = best_successor

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
