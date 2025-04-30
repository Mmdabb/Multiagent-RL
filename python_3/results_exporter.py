import pandas as pd

def export_agent_results(agents, output_path):
    """
    Export agent-level results to CSV.

    Args:
        agents (list of dict): Each agent must have 'agent_id', 'origin_node', 'destination_node', and 'traveled_path'.
        output_path (str): Output CSV path.
    """
    records = []
    for agent in agents:
        records.append({
            'agent_id': agent['agent_id'],
            'origin_node': agent['origin_node'],
            'destination_node': agent['destination_node'],
            'link_sequence': agent.get('traveled_path', [])
        })

    df = pd.DataFrame(records)

    # Convert list to space-separated string
    df['link_sequence'] = df['link_sequence'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else '')

    df.to_csv(output_path, index=False)
    print(f" Agent results exported to {output_path}")


def export_link_performance(G, output_file: str):
    """
    Export link-level flow and travel time results from the network graph.

    Args:
        G (nx.DiGraph): Network graph with 'link_id', 'flow', and 'current_travel_time' on each edge.
        output_file (str): Output CSV file path.
    """
    records = []

    for u, v, attr in G.edges(data=True):
        link_id = attr.get('link_id')
        flow = attr.get('flow', 0)
        travel_time = attr.get('current_travel_time', attr.get('free_flow_travel_time'))

        records.append({
            'link_id': link_id,
            'from_node_id': u,
            'to_node_id': v,
            'flow': flow,
            'travel_time': travel_time,
            'free_flow_travel_time': attr.get('free_flow_travel_time')
        })

    df = pd.DataFrame(records)
    df.sort_values(by='link_id', inplace=True)
    df.to_csv(output_file, index=False)
    print(f" Link performance exported to {output_file}")
