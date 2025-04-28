import networkx as nx
import pandas as pd


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



def aggregate_agent_link_flows(agent_result_file, output_file):
    """
    Aggregate link flows from agent paths and save to CSV.

    Args:
        agent_result_file (str): Path to agent results CSV.
        output_file (str): Path to save aggregated link flow CSV.
    """
    df = pd.read_csv(agent_result_file)

    link_flow_count = {}

    for links in df['link_sequence']:
        if pd.isna(links):
            continue

        links = str(links).strip()

        if links.startswith('[') and links.endswith(']'):
            # Proper list format
            link_ids = eval(links)
        else:
            # Bad format, try to recover: split by spaces
            link_ids = [int(x) for x in links.split() if x.strip().isdigit()]

        for link_id in link_ids:
            link_flow_count[link_id] = link_flow_count.get(link_id, 0) + 1

    # Save to DataFrame
    flow_df = pd.DataFrame({
        'link_id': list(link_flow_count.keys()),
        'flow_from_agent_paths': list(link_flow_count.values())
    })

    flow_df = flow_df.sort_values('link_id').reset_index(drop=True)
    flow_df.to_csv(output_file, index=False)

    print(f"Aggregated agent path flows saved to {output_file}")
