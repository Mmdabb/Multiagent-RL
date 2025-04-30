# network_loader.py

import pandas as pd
import networkx as nx


def load_network(node_file: str, link_file: str):
    """
    Load nodes and links from CSV files and create a directed graph.

    Args:
        node_file (str): Path to the node CSV file.
        link_file (str): Path to the link CSV file.

    Returns:
        G (nx.DiGraph): Directed networkx graph with node and link attributes.
    """
    # Load node and link data
    node_df = pd.read_csv(node_file)
    link_df = pd.read_csv(link_file)

    # Initialize directed graph
    G = nx.DiGraph()

    # Add nodes
    for _, row in node_df.iterrows():
        zone_id = row['zone_id'] if pd.notna(row['zone_id']) and row['zone_id'] != 0 else None
        G.add_node(row['node_id'],
                   x_coord=row['x_coord'],
                   y_coord=row['y_coord'],
                   zone_id=zone_id)

    # Add links
    for _, row in link_df.iterrows():
        total_capacity = row['lanes'] * row['capacity']
        free_flow_travel_time = (row['length'] / row['free_speed']) * 60 / 1000  # minutes

        G.add_edge(row['from_node_id'], row['to_node_id'],
                   link_id=row['link_id'],
                   length=row['length'],
                   lanes=row['lanes'],
                   free_speed=row['free_speed'],
                   capacity=total_capacity,
                   free_flow_travel_time=free_flow_travel_time)

    return G
