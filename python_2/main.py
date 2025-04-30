import os
import numpy as np
from python_3.config import *
from python_3.network_loader import load_network
from python_3.demand_loader import load_demand
from python_3.value_function_solver import solve_value_function
from path_assignment import assign_initial_paths
from one_step_rollout import run_one_step_multiagent_rollout
from python_3.results_exporter import export_agent_results, export_link_performance




def initialize_link_travel_times(G):
    for u, v, attr in G.edges(data=True):
        attr['flow'] = 0
        attr['current_travel_time'] = attr['free_flow_travel_time']


def main():
    # === Step 0: Load Inputs ===

    G = load_network(node_file, link_file)

    agents, destination_zones = load_demand(demand_file)

    # === Step 1: Initialize Costs ===
    initialize_link_travel_times(G)

    # === Step 2: Solve Initial Value Functions (Base Policy) ===
    value_function_dict = {}
    for dest_zone in destination_zones:
        value_function = solve_value_function(G, dest_zone)
        value_function_dict[dest_zone] = value_function

    # === Step 3: Assign Initial Paths to Agents ===
    assign_initial_paths(G, agents, value_function_dict, method='shortest_path')

    link_flows = {attr['link_id']: 0 for _, _, attr in G.edges(data=True)}
    system_travel_time_history = []
    outer_iter = 0

    # === Step 4: Outer Rollout Iterations ===
    # for outer_iter in range(max_outer_iterations):
    while True:

        print(f"\n=== Rollout Iteration {outer_iter + 1} ===")

        # One-step rollout per agent
        completed_count = run_one_step_multiagent_rollout(G, agents, link_flows, value_function_dict, greedy=False, mu=mu, random_seed=random_seed)

        if completed_count == len(agents):
            print(f"\n All agents completed their trips by iteration {outer_iter + 1}.")
            break

        # Recompute value function based on updated costs
        for dest_zone in destination_zones:
            value_function = solve_value_function(G, dest_zone)
            value_function_dict[dest_zone] = value_function

        total_system_tt = sum(
            attr['current_travel_time'] * link_flows.get(attr['link_id'], 0)
            for _, _, attr in G.edges(data=True)
        )
        system_travel_time_history.append(total_system_tt)

        # (Optional: Add convergence tracking here)

        outer_iter += 1

    # === Step 5: Export Results ===
    export_agent_results(agents, agent_result_file)
    export_link_performance(G, link_flows, link_performance_file)
    np.savetxt(os.path.join(data_path, "multiagent_system_travel_time.csv"), system_travel_time_history, delimiter=",")

    print("\n Simulation complete. Results exported.")
if __name__ == "__main__":
    main()
