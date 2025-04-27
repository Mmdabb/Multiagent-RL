import os
from network_loader import load_network
from demand_loader import load_demand
from value_function_solver import solve_value_function
from multi_agent_rollout import multi_agent_rollout
from results_exporter import export_agent_results, export_link_performance


def main():
    # File paths
    data_path = "../data_sets/3-corridor"
    node_file = os.path.join(data_path, "node.csv")
    link_file = os.path.join(data_path, "link.csv")
    demand_file = os.path.join(data_path, "demand.csv")

    agent_result_file = os.path.join(data_path, 'agent_result.csv')
    link_performance_file = os.path.join(data_path, 'link_performance.csv')

    # Step 1: Load network
    G = load_network(node_file, link_file)
    print(f"Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Step 2: Load demand
    agents, destination_zones = load_demand(demand_file)
    print(f"Loaded {len(agents)} agents.")

    # Step 3: Solve value function
    value_function = solve_value_function(G, destination_zones)

    # Step 4: Multi-agent rollout
    agent_paths, link_flows = multi_agent_rollout(G, agents, value_function)

    # Step 5: Export results
    export_agent_results(agent_paths, agent_result_file)
    export_link_performance(G, link_flows, link_performance_file)


if __name__ == "__main__":
    main()
