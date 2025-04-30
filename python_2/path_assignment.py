import networkx as nx
import heapq

def assign_initial_paths(G, agents, value_function_dict, method='shortest_path'):
    """
    Assign an initial path to each agent using a base policy.
    Currently supports shortest path assignment based on the value function.

    Parameters:
    - G (nx.DiGraph): The network graph.
    - agents (list of dict): Each agent has 'agent_id', 'origin_node', 'destination_node'.
    - value_function_dict (dict): Maps destination zone to value function.
    - method (str): Assignment method. Currently only 'shortest_path' is implemented.

    Returns:
    - agents: Modified with assigned 'path' (link sequence), 'current_position', and path memory.
    """
    for agent in agents:
        origin = agent['origin_node']
        dest = agent['destination_node']
        V = value_function_dict[dest]

        if method == 'shortest_path':
            # Dijkstra-like greedy forward assignment using V
            current = origin
            path_links = []
            while current != dest:
                successors = list(G.successors(current))
                if not successors:
                    break
                best_next = None
                best_score = float('-inf')
                for succ in successors:
                    score = -G[current][succ]['current_travel_time'] + V.get(succ, float('-inf'))
                    if score > best_score:
                        best_score = score
                        best_next = succ
                if best_next is None:
                    break
                path_links.append(G[current][best_next]['link_id'])
                current = best_next

            agent['path'] = path_links
            agent['traveled_path'] = []
            agent['current_position'] = origin
            agent['previous_link_id'] = None

        else:
            raise NotImplementedError(f"Assignment method '{method}' not implemented yet.")

    return agents
