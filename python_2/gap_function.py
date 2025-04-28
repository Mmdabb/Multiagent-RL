import networkx as nx

def compute_max_path_gap(G, agents):
    """
    Compute the maximum path gap across all agents.

    Args:
        G (nx.DiGraph): Network graph (with current travel times).
        agents (list): List of agent path dictionaries from rollout.

    Returns:
        max_gap (float): Maximum gap (assigned - shortest path cost).
        avg_gap (float): Average gap across all agents.
    """
    max_gap = 0.0
    total_gap = 0.0
    valid_agents = 0

    # Use current travel time
    travel_time_attr = {
        (u, v): attr.get('current_travel_time', attr['free_flow_travel_time'])
        for u, v, attr in G.edges(data=True)
    }

    for agent in agents:
        origin = agent['o_zone_id']
        destination = agent['d_zone_id']
        assigned_cost = agent['path_travel_time']

        try:
            # Compute shortest path cost using current travel times
            shortest_path_length = nx.shortest_path_length(
                G,
                source=origin,
                target=destination,
                weight=lambda u, v, d: d.get('current_travel_time', d['free_flow_travel_time'])
            )
        except nx.NetworkXNoPath:
            print(f"Warning: No path found for agent {agent['agent_id']} from {origin} to {destination}")
            continue

        gap = assigned_cost - shortest_path_length
        if gap < 0:
            gap = 0  # Numerical tolerance: assigned might be slightly better due to rounding

        total_gap += gap
        if gap > max_gap:
            max_gap = gap
        valid_agents += 1

    avg_gap = total_gap / valid_agents if valid_agents > 0 else 0.0

    return max_gap, avg_gap
