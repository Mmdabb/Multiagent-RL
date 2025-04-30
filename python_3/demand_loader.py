import pandas as pd


def load_demand(demand_file: str):
    """
    Load demand from CSV file and create a list of agent dictionaries.

    Returns:
        agents (list): List of agent dictionaries.
        destination_zones (set): Set of unique destination zone IDs.
    """
    demand_df = pd.read_csv(demand_file)

    agents = []
    destination_zones = set(demand_df['d_zone_id'].unique())
    agent_id_counter = 0

    for _, row in demand_df.iterrows():
        for _ in range(int(row['volume'])):
            agent = {
                'agent_id': agent_id_counter,
                'origin_node': row['o_zone_id'],
                'destination_node': row['d_zone_id']
            }
            agents.append(agent)
            agent_id_counter += 1

    return agents, destination_zones
