from utils import update_link_cost_bpr



def assign_paths_from_value_function(G, agents, value_function_dict):
    """
    Assign initial greedy path to each agent using the value function.

    Parameters:
    - G: networkx DiGraph
    - agents: list of agent dicts
    - value_function_dict: {dest_zone: {node: value}}

    Updates:
    - Sets each agent['planned_links'] with full greedy path
    - Increments flow along those links
    """

    for agent in agents:
        origin = agent['origin_node']
        dest = agent['destination_node']
        V = value_function_dict[dest]

        path = []
        curr = origin

        while curr != dest:
            successors = list(G.successors(curr))
            if not successors:
                break

            best_succ = max(successors, key=lambda s: -G[curr][s]['current_travel_time'] + V.get(s, float('-inf')))
            path.append((curr, best_succ))
            G[curr][best_succ]['flow'] += 1  # Initial flow increment
            curr = best_succ

        agent['current_position'] = origin
        agent['traveled_path'] = []
        agent['planned_links'] = path

        for u, v, attr in G.edges(data=True):
            update_link_cost_bpr(G, u, v)


