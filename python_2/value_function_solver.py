# value_function_solver.py

import networkx as nx
import numpy as np

def solve_value_function(G: nx.DiGraph, destination_zones: set, mu: float = 1.0):
    """
    Solve the soft Bellman equation for the Recursive Logit model.

    Args:
        G (nx.DiGraph): Network graph.
        destination_zones (set): Set of valid destination zone IDs.
        mu (float): Softmax temperature parameter.

    Returns:
        value_function (dict): Dictionary {node_id: V(node)}.
    """
    value_function = {}

    # Step 1: Identify true destination nodes based on zone_id and demand
    destination_nodes = [
        n for n, attr in G.nodes(data=True)
        if attr.get('zone_id') in destination_zones
    ]

    # Step 2: Initialize values
    for node in G.nodes():
        value_function[node] = 0.0 if node in destination_nodes else np.inf

    # Step 3: Value iteration backward
    convergence_threshold = 1e-4
    max_iterations = 1000

    for iteration in range(max_iterations):
        delta = 0
        updated_value = value_function.copy()

        for node in G.nodes():
            if node in destination_nodes:
                continue

            scores = []
            for _, successor, attr in G.out_edges(node, data=True):
                score = attr['free_flow_travel_time'] + value_function[successor]
                scores.append(-score / mu)  # Negative because smaller travel time is better

            if scores:
                updated_value[node] = -mu * np.log(np.sum(np.exp(scores)))
                delta = max(delta, abs(updated_value[node] - value_function[node]))

        value_function = updated_value

        if delta < convergence_threshold:
            print(f"Value function converged in {iteration+1} iterations.")
            break

    return value_function
