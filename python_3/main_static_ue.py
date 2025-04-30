import os
import numpy as np
import pandas as pd
import networkx as nx
from config import *
from network_loader import load_network
from demand_loader import load_demand
from utils import update_link_travel_times
import time

def main():
    # Load network and demand
    start_time = time.time()

    G = load_network(node_file, link_file)
    agents, destination_zones = load_demand(demand_file)

    # Initialize free-flow travel times
    for u, v, attr in G.edges(data=True):
        attr['current_travel_time'] = attr['free_flow_travel_time']

    flow_change_history = []
    system_travel_time_history = []
    last_link_flows = None

    for outer_iter in range(max_outer_iterations):
        print(f"=== Static UE Iteration {outer_iter+1} ===")

        # Step 1: All-Or-Nothing assignment
        new_link_flows = {attr['link_id']: 0 for _, _, attr in G.edges(data=True)}

        for agent in agents:
            try:
                shortest_path = nx.shortest_path(
                    G,
                    source=agent['origin_node'],
                    target=agent['destination_node'],
                    weight=lambda u, v, d: d.get('current_travel_time', d['free_flow_travel_time'])
                )
            except nx.NetworkXNoPath:
                continue

            # Increment flows along the path
            for i in range(len(shortest_path) - 1):
                u = shortest_path[i]
                v = shortest_path[i+1]
                link_id = G[u][v]['link_id']
                new_link_flows[link_id] += 1  # 1 trip per agent

        # Step 2: MSA Flow Update
        if last_link_flows is None:
            relaxed_link_flows = new_link_flows.copy()
        else:
            lambda_relax = 1.0 / (outer_iter + 1)
            relaxed_link_flows = {}
            for link_id in new_link_flows.keys():
                old_flow = last_link_flows.get(link_id, 0)
                new_flow = new_link_flows[link_id]
                relaxed_flow = (1 - lambda_relax) * old_flow + lambda_relax * new_flow
                relaxed_link_flows[link_id] = relaxed_flow

        # Step 3: Update travel times
        update_link_travel_times(G, relaxed_link_flows)

        # Step 4: Log system travel time
        total_system_tt = sum(
            attr['current_travel_time'] * relaxed_link_flows.get(attr['link_id'], 0)
            for _, _, attr in G.edges(data=True)
        )
        system_travel_time_history.append(total_system_tt)



        last_link_flows = relaxed_link_flows.copy()

        # Convergence check (max flow change)
        # if last_link_flows is not None:
        #     max_flow_change = max(
        #         abs(relaxed_link_flows[link_id] - last_link_flows.get(link_id, 0))
        #         for link_id in relaxed_link_flows
        #     )
        #     print(f"Max link flow change: {max_flow_change:.6f}")
        #     if max_flow_change < 1e-4:
        #         print(f"Converged after {outer_iter + 1} iterations (flow change)")
        #         break

        # Optional: relative gap in total travel time
        if outer_iter > 0:
            rel_gap = abs(system_travel_time_history[-1] - system_travel_time_history[-2]) / system_travel_time_history[
                -2]
            print(f"Relative system travel time gap: {rel_gap:.6f}")
            if rel_gap < 1e-3:
                print(f"Converged after {outer_iter + 1} iterations (relative gap)")
                break

    end_time = time.time()
    print(f"Computation time: {end_time - start_time:.4f} seconds")

    records = []
    for u, v, attr in G.edges(data=True):
        link_id = attr['link_id']
        flow = last_link_flows.get(link_id, 0)
        # travel_time = attr['free_flow_travel_time']  # still using free flow for now
        travel_time = attr.get('current_travel_time', attr['free_flow_travel_time'])

        records.append({
            'link_id': link_id,
            'from_node_id': u,
            'to_node_id': v,
            'flow': flow,
            'travel_time': travel_time,
            'free_flow_travel_time': attr['free_flow_travel_time']
        })

    df_link_performance = pd.DataFrame(records)
    # df_link_performance.sort_values(by='link_id', ascending=True)

    df_link_performance.to_csv(os.path.join(data_path, "UE_linkperformance.csv"), index=False)

    # Save static assignment history
    np.savetxt(os.path.join(data_path, "static_ue_system_travel_time.csv"), system_travel_time_history, delimiter=",")

if __name__ == "__main__":
    main()
