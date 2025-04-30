import numpy as np
from python_3.helper_functions import softmax, update_link_cost_bpr  # assumes you have a softmax utility

def run_one_step_multiagent_rollout(G, agents, link_flows, value_function_dict, greedy=False, mu=0.1, random_seed=None):
    """
    Performs one-step rollout per agent in random order, allowing agents to deviate from assigned path.

    Parameters:
    - G: The network graph with 'current_travel_time' and 'flow' on edges.
    - agents: List of agent dicts with path, position, etc.
    - value_function_dict: Map from destination to node-wise value function.
    - mu: Softmax temperature.

    Returns:
    - None (agents and G are modified in-place).
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    agent_indices = np.random.permutation(len(agents))
    completed_count = 0
    for idx in agent_indices:
        agent = agents[idx]
        aid = agent['agent_id']
        curr = agent['current_position']
        dest = agent['destination_node']
        if curr == dest:
            completed_count += 1
            continue

        V = value_function_dict[dest]
        candidates = []
        scores = []

        for succ in G.successors(curr):
            attr = G[curr][succ]
            link_cost = attr['current_travel_time']
            succ_value = V.get(succ, float('-inf'))
            score = -link_cost + succ_value
            candidates.append((curr, succ))
            scores.append(score)

        if not candidates:
            continue

        if greedy:
            # Pick highest score
            selected_idx = int(np.argmax(scores))
        else:
            # Softmax sampling
            probs = softmax(np.array(scores) / mu)
            selected_idx = np.random.choice(len(candidates), p=probs)


        from_node, to_node = candidates[selected_idx]
        link_id = G[from_node][to_node]['link_id']

        # Move agent
        agent['traveled_path'].append(link_id)
        agent['current_position'] = to_node

        # Update flows
        G[from_node][to_node]['flow'] += 1
        link_flows[link_id] += 1
        if agent['previous_link_id'] is not None:
            prev_from, prev_to = agent['previous_link_id']
            G[prev_from][prev_to]['flow'] -= 1
        agent['previous_link_id'] = (from_node, to_node)

        # Update travel time using BPR
        update_link_cost_bpr(G, from_node, to_node)

    return completed_count