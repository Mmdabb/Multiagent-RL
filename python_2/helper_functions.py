import networkx as nx

def find_dead_end_nodes(G: nx.DiGraph, destination_zones: set):
    """
    Find nodes that have no outgoing links and are not destinations.

    Args:
        G (nx.DiGraph): Network graph.
        destination_zones (set): Set of valid destination zone IDs.

    Returns:
        dead_ends (list): List of dead-end node IDs.
    """
    dead_ends = []
    for node in G.nodes():
        is_destination = G.nodes[node].get('zone_id') in destination_zones
        has_outgoing_links = G.out_degree(node) > 0

        if not has_outgoing_links and not is_destination:
            dead_ends.append(node)

    return dead_ends
