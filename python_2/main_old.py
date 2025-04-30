import os
import numpy as np
from python_3.config import *
from python_3.network_loader import load_network
from python_3.demand_loader import load_demand
from python_3.value_function_solver import solve_value_function
from stochastic_multi_agent_rollout import multi_agent_rollout
from python_3.helper_functions import update_link_travel_times, aggregate_agent_link_flows
from python_3.results_exporter import export_agent_results, export_link_performance
from gap_function import compute_max_path_gap


def main():
    # Load network and demand
    G = load_network(node_file, link_file)
    agents, destination_zones = load_demand(demand_file)

    # Initialize free-flow travel times as current travel times
    for u, v, attr in G.edges(data=True):
        attr['current_travel_time'] = attr['free_flow_travel_time']

    system_travel_time_history = []
    last_link_flows = None

    for outer_iter in range(max_outer_iterations):
        print(f"=== Outer Iteration {outer_iter+1} ====================================================")

        # Step 1: Solve value functions for each destination
        value_function_dict = {}
        for dest_zone in destination_zones:
            value_function = solve_value_function(G, dest_zone)
            value_function_dict[dest_zone] = value_function

        # Step 2: Multi-agent rollout based on current value functions
        # agent_paths, new_link_flows = multi_agent_rollout(G, agents, value_function_dict)
        agent_paths, new_link_flows = multi_agent_rollout(G, agents, value_function_dict, mu=0.1, random_seed=42)

        # Step 3: Relaxation (flow averaging)
        if last_link_flows is None:
            relaxed_link_flows = new_link_flows.copy()
        else:
            if use_msa:
                lambda_relax = 1.0 / (outer_iter + 1)  # MSA relaxation
            else:
                lambda_relax = 1.0  # NO relaxation (fully update to new flow)

            relaxed_link_flows = {}
            for link_id in new_link_flows.keys():
                old_flow = last_link_flows.get(link_id, 0)
                new_flow = new_link_flows[link_id]
                relaxed_flow = (1 - lambda_relax) * old_flow + lambda_relax * new_flow
                relaxed_link_flows[link_id] = relaxed_flow

        # Step 4: Update travel times using relaxed flows (BPR)
        update_link_travel_times(G, relaxed_link_flows)

        total_system_tt = sum(
            attr['current_travel_time'] * new_link_flows.get(attr['link_id'], 0)
            for _, _, attr in G.edges(data=True)
        )
        system_travel_time_history.append(total_system_tt)

        # Step 5: Convergence checking
        if last_link_flows is not None:
            max_flow_change = max(
                abs(relaxed_link_flows.get(link_id, 0) - last_link_flows.get(link_id, 0))
                for link_id in relaxed_link_flows.keys()
            )
            print(f"Max link flow change: {max_flow_change:.6f}")

            if max_flow_change < convergence_threshold:
                print(f"Converged after {outer_iter+1} iterations.")
                break

        # Update last_link_flows for next iteration
        last_link_flows = relaxed_link_flows.copy()
        # Compute UE condition
        max_gap, avg_gap = compute_max_path_gap(G, agent_paths)
        print(f"Max Path Gap: {max_gap:.6f} minutes, Avg Path Gap: {avg_gap:.6f} minutes")

    # Final step: Export final results
    export_agent_results(agent_paths, agent_result_file)
    export_link_performance(G, new_link_flows, link_performance_file)
    aggregate_agent_link_flows(agent_result_file, agent_link_flow_output)

    np.savetxt(os.path.join(data_path, "multiagent_system_travel_time.csv"), system_travel_time_history, delimiter=",")


if __name__ == "__main__":
    main()
