# main.py
import os
from network_loader import load_network
from demand_loader import load_demand
from value_function_solver import solve_value_function
from multi_agent_rollout import multi_agent_rollout
from travel_time_updater import update_link_travel_times
from results_exporter import export_agent_results, export_link_performance
from gap_function import compute_max_path_gap

def main():
    # File paths
    data_path = "../data_sets/toy"
    node_file = os.path.join(data_path, "node.csv")
    link_file = os.path.join(data_path, "link.csv")
    demand_file = os.path.join(data_path, "demand.csv")

    agent_result_file = os.path.join(data_path, 'agent_result.csv')
    link_performance_file = os.path.join(data_path, 'link_performance.csv')

    # Load network and demand
    G = load_network(node_file, link_file)
    agents, destination_zones = load_demand(demand_file)

    # Initialize free-flow travel times as current travel times
    for u, v, attr in G.edges(data=True):
        attr['current_travel_time'] = attr['free_flow_travel_time']

    max_outer_iterations = 50  # You can increase a bit since convergence will be smoother
    convergence_threshold = 1e-3  # Threshold for max flow change

    last_link_flows = None

    for outer_iter in range(max_outer_iterations):
        print(f"=== Outer Iteration {outer_iter+1} ====================================================")

        # Step 1: Solve value functions for each destination
        value_function_dict = {}
        for dest_zone in destination_zones:
            value_function = solve_value_function(G, dest_zone)
            value_function_dict[dest_zone] = value_function

        # Step 2: Multi-agent rollout based on current value functions
        agent_paths, new_link_flows = multi_agent_rollout(G, agents, value_function_dict)

        # Step 3: Relaxation (flow averaging)
        if last_link_flows is None:
            # First iteration: use new flows directly
            relaxed_link_flows = new_link_flows.copy()
        else:
            lambda_relax = 1.0 / (outer_iter + 1)  # MSA relaxation
            relaxed_link_flows = {}
            for link_id in new_link_flows.keys():
                old_flow = last_link_flows.get(link_id, 0)
                new_flow = new_link_flows[link_id]
                relaxed_flow = (1 - lambda_relax) * old_flow + lambda_relax * new_flow
                relaxed_link_flows[link_id] = relaxed_flow

        # Step 4: Update travel times using relaxed flows (BPR)
        update_link_travel_times(G, relaxed_link_flows)

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
    export_link_performance(G, relaxed_link_flows, link_performance_file)

if __name__ == "__main__":
    main()
