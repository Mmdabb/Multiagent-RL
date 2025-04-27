# results_exporter.py

import pandas as pd

def export_agent_results(agent_paths: list, output_file: str):
    """
    Export agent path results to CSV.

    Args:
        agent_paths (list): List of agent path dictionaries.
        output_file (str): Output file path.
    """
    df = pd.DataFrame(agent_paths)
    # Convert sequences to string for CSV writing
    df['link_sequence'] = df['link_sequence'].apply(lambda x: ' '.join(map(str, x)))
    df['node_sequence'] = df['node_sequence'].apply(lambda x: ' '.join(map(str, x)))
    df.to_csv(output_file, index=False)
    print(f"Agent results exported to {output_file}")

def export_link_performance(G, link_flows: dict, output_file: str):
    """
    Export link-level performance statistics to CSV.

    Args:
        G (nx.DiGraph): Network graph.
        link_flows (dict): Dictionary {link_id: total flow assigned}.
        output_file (str): Output file path.
    """
    records = []

    for u, v, attr in G.edges(data=True):
        link_id = attr['link_id']
        flow = link_flows.get(link_id, 0)
        travel_time = attr['free_flow_travel_time']  # still using free flow for now

        records.append({
            'link_id': link_id,
            'from_node_id': u,
            'to_node_id': v,
            'flow': flow,
            'travel_time': travel_time,
            'free_flow_travel_time': attr['free_flow_travel_time']
        })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Link performance results exported to {output_file}")
