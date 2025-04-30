import os
import numpy as np
from network_loader import load_network
from demand_loader import load_demand
from value_function_solver import solve_value_function
from path_assignment import assign_paths_from_value_function
from one_step_rollout_replanned import run_one_step_multiagent_rollout
from results_exporter import export_agent_results, export_link_performance
from utils import update_link_cost_bpr, make_rollout_update_function
from config import *
import copy
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import make_rollout_update_function
import time


def initialize_travel_times(G):
    for u, v, attr in G.edges(data=True):
        attr['flow'] = 0
        attr['current_travel_time'] = attr['free_flow_travel_time']

def main():
    # Load data
    start_time = time.time()

    G = load_network(node_file, link_file)
    agents, destination_zones = load_demand(demand_file)
    initialize_travel_times(G)

    # Initial value function & path assignment
    value_function_dict = {}
    for dest_zone in destination_zones:
        value_function = solve_value_function(G, dest_zone)
        value_function_dict[dest_zone] = value_function

    assign_paths_from_value_function(G, agents, value_function_dict)
    export_link_performance(G, os.path.join(data_path, "initial_link_performance.csv"))

    for dest_zone in destination_zones:
        value_function = solve_value_function(G, dest_zone)
        value_function_dict[dest_zone] = value_function

    system_travel_time_history = []  # Track system travel time

    # Outer iteration loop
    outer_iter = 0
    best_tt = 0
    while outer_iter < max_outer_iterations:
        print(f"\n=== Rollout Iteration {outer_iter + 1} ===")

        # ✅ Add this block
        total_tt = sum(
            attr['current_travel_time'] * attr.get('flow', 0)
            for _, _, attr in G.edges(data=True)
        )

        if total_tt < best_tt or not best_tt:
            best_tt = total_tt

        system_travel_time_history.append(best_tt)

        completed_count = run_one_step_multiagent_rollout(
            G, agents, value_function_dict, mu=mu, greedy=False
        )

        # Create an output folder once
        os.makedirs(os.path.join(data_path, "snapshots"), exist_ok=True)

        # Save a deep copy of G and agents
        snapshot = {
            'G': copy.deepcopy(G),
            'agents': copy.deepcopy(agents)
        }
        with open(os.path.join(data_path, f"snapshots/iter_{outer_iter}.pkl"), 'wb') as f:
            pickle.dump(snapshot, f)

        for u, v in G.edges():
            update_link_cost_bpr(G, u, v)



        export_link_performance(G, os.path.join(data_path, f"link_performance_iter{outer_iter}.csv"))
        # if completed_count == len(agents):
        #     print(f"\n All agents completed by iteration {outer_iter + 1}")
        #     break

        # Update value function for all destinations
        for dest_zone in destination_zones:
            value_function = solve_value_function(G, dest_zone)
            value_function_dict[dest_zone] = value_function

        outer_iter += 1

    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.4f} seconds")
    # Export results
    export_agent_results(agents, agent_result_file)
    export_link_performance(G, link_performance_file)
    np.savetxt(os.path.join(data_path, "multiagent_system_travel_time.csv"),
               system_travel_time_history, delimiter=",")

    print("\n Export complete.")



    # === Load snapshots ===
    snapshot_folder = os.path.join(data_path, 'snapshots')
    snapshot_files = sorted(os.listdir(snapshot_folder))
    snapshots = []
    for filename in snapshot_files:
        with open(os.path.join(snapshot_folder, filename), 'rb') as f:
            snapshots.append(pickle.load(f))

    # === Prepare layout and agent sampling ===
    G0 = snapshots[0]['G']
    # pos = nx.spring_layout(G0, seed=42)  # Or based on actual coordinates
    # pos = {node: (data['x_coord'], data['y_coord']) for node, data in G.nodes(data=True)}

    pos = {
        1: (0, 0),
        2: (3, 0.2),
        3: (5, 0.2),
        4: (4, -0.2),
        5: (8, 0.2),
        6: (8, -0.2),
    }

    all_agents = snapshots[0]['agents']
    agent_ids = random.sample([a['agent_id'] for a in snapshots[0]['agents']], 4)  # 🎯 Only 2 agents

    fig, ax = plt.subplots(figsize=(10, 8))
    update_fn = make_rollout_update_function(snapshots, pos, agent_ids, ax)

    ani = animation.FuncAnimation(fig, update_fn, frames=len(snapshots), interval=1000, repeat=False)
    # ani.save(os.path.join(data_path, "rollout_animation.gif"), writer="pillow")
    for i, snapshot in enumerate(snapshots):
        fig, ax = plt.subplots()
        update_fn(i)  # call your update() function directly
        plt.savefig(os.path.join(data_path,f"frame_{i:03d}.png"))
        plt.close()
    # ani.save(os.path.join(data_path,"rollout_animation.mp4"), writer="pillow", fps=1)  # or fps=2 for faster playback

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
