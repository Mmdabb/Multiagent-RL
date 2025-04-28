import networkx as nx
import numpy as np


def solve_value_function(G: nx.DiGraph, destination_zone: int, mu: float = 1.0):
    """
    Solve the soft Bellman equation for a single destination.

    Args:
        G (nx.DiGraph): Network graph.
        destination_zone (int): Zone ID of the destination.
        mu (float): Softmax temperature parameter.

    Returns:
        value_function (dict): Dictionary {node_id: V(node)} for this destination.
    """
    value_function = {}

    # Identify the destination node corresponding to this zone
    destination_nodes = [
        n for n, attr in G.nodes(data=True)
        if attr.get('zone_id') == destination_zone
    ]

    if len(destination_nodes) != 1:
        raise ValueError(f"Destination zone {destination_zone} does not match exactly one node.")

    destination_node = destination_nodes[0]

    # Initialize
    for node in G.nodes():
        if node == destination_node:
            value_function[node] = 0.0
        else:
            value_function[node] = -np.inf

    # Standard Value Iteration (same as fixed version before)
    convergence_threshold = 1e-4
    max_iterations = 1000

    for iteration in range(max_iterations):
        delta = 0
        updated_value = value_function.copy()

        for node in G.nodes():
            if node == destination_node:
                continue

            print(f"Node {node}: outgoing successors: {[succ for _, succ in G.out_edges(node)]}")
            scores = []
            for _, successor, attr in G.out_edges(node, data=True):
                # link_cost = attr['free_flow_travel_time']
                link_cost = attr.get('current_travel_time', attr['free_flow_travel_time'])
                utility = -link_cost
                total_score = utility + value_function[successor]
                scores.append(total_score / mu)
                print(f"Node {node}; to node:{successor}; link_id:{attr.get('link_id')}; link cost:{link_cost:.4f}; value function successor: {value_function[successor]:.4f}")


            # print()
            if scores and not all(np.isneginf(scores)):
                updated_value[node] = mu * np.log(np.sum(np.exp(scores)))
            else:
                updated_value[node] = -np.inf

            delta = max(delta, abs(updated_value[node] - value_function[node]))
            # print(f"Node {node}: outgoing successors: {[succ for _, succ in G.out_edges(node)]}")
            if node == 2:
                print(f"iteration: {iteration}; node:{node}; value_function: {updated_value[node]}; scores: {scores}")

        value_function = updated_value


        if delta < convergence_threshold:
            print(f"Value function for destination {destination_zone} converged in {iteration+1} iterations (Δ={delta:.6f}).")
            # print(f"After iteration {iteration + 1}:")
            for node, val in value_function.items():
                print(f"V({node}) = {val:.4f}")
            print("\n")
            break

    return value_function
