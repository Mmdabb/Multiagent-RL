# travel_time_updater.py

def update_link_travel_times(G, link_flows, alpha=0.15, beta=4):
    """
    Update link travel times using the BPR function.

    Args:
        G (nx.DiGraph): Network graph.
        link_flows (dict): {link_id: flow}.
        alpha (float): BPR alpha parameter.
        beta (float): BPR beta parameter.
    """
    for u, v, attr in G.edges(data=True):
        link_id = attr['link_id']
        flow = link_flows.get(link_id, 0)
        free_flow_time = attr['free_flow_travel_time']
        capacity = attr['capacity']

        updated_time = free_flow_time * (1 + alpha * (flow / capacity) ** beta)
        attr['current_travel_time'] = updated_time
