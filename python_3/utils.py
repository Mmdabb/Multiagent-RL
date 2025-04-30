import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def trace_greedy_path_from_value_function(G, start_node, destination_node, value_function):
    """
    Trace a greedy shortest path from current node to destination using value function.
    """
    curr = start_node
    path = []
    while curr != destination_node:
        successors = list(G.successors(curr))
        if not successors:
            break
        best_succ = max(successors, key=lambda s: -G[curr][s]['current_travel_time'] + value_function.get(s, float('-inf')))
        path.append((curr, best_succ))
        curr = best_succ
    return path




def find_dead_end_nodes(G: nx.DiGraph, destination_zones: set):
    """
    Find nodes that have no outgoing links and are not destinations.

    Args:
        G (nx.DiGraph): Network graph.
        destination_zones (set): Set of valid destination zone IDs.

    Returns:
        dead_ends (list): List of dead-end node IDs.
    """
    dead_ends = []
    for node in G.nodes():
        is_destination = G.nodes[node].get('zone_id') in destination_zones
        has_outgoing_links = G.out_degree(node) > 0

        if not has_outgoing_links and not is_destination:
            dead_ends.append(node)

    return dead_ends



def aggregate_agent_link_flows(agent_result_file, output_file):
    """
    Aggregate link flows from agent paths and save to CSV.

    Args:
        agent_result_file (str): Path to agent results CSV.
        output_file (str): Path to save aggregated link flow CSV.
    """
    df = pd.read_csv(agent_result_file)

    link_flow_count = {}

    for links in df['link_sequence']:
        if pd.isna(links):
            continue

        links = str(links).strip()

        if links.startswith('[') and links.endswith(']'):
            # Proper list format
            link_ids = eval(links)
        else:
            # Bad format, try to recover: split by spaces
            link_ids = [int(x) for x in links.split() if x.strip().isdigit()]

        for link_id in link_ids:
            link_flow_count[link_id] = link_flow_count.get(link_id, 0) + 1

    # Save to DataFrame
    flow_df = pd.DataFrame({
        'link_id': list(link_flow_count.keys()),
        'flow_from_agent_paths': list(link_flow_count.values())
    })

    flow_df = flow_df.sort_values('link_id').reset_index(drop=True)
    flow_df.to_csv(output_file, index=False)

    print(f"Aggregated agent path flows saved to {output_file}")



def update_link_travel_times(G, link_flows, alpha=0.15, beta=4):
    """
    Update link travel times using the BPR function.

    Args:
        G (nx.DiGraph): Network graph.
        link_flows (dict): {link_id: flow}.
        alpha (float): BPR alpha parameter.
        beta (float): BPR beta parameter.
    """
    for u, v, attr in G.edges(data=True):
        link_id = attr['link_id']
        flow = link_flows.get(link_id, 0)
        free_flow_time = attr['free_flow_travel_time']
        capacity = attr['capacity']

        updated_time = free_flow_time * (1 + alpha * (flow / capacity) ** beta)
        attr['current_travel_time'] = updated_time



def update_link_cost_bpr(G, u, v, alpha=0.15, beta=4):
    attr = G[u][v]
    t0 = attr['free_flow_travel_time']
    flow = attr.get('flow', 0)
    capacity = attr['capacity']
    attr['current_travel_time'] = t0 * (1 + alpha * (flow / capacity) ** beta)


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_rollout_update_function(snapshots, pos, agent_ids, ax):
    """
    Creates a matplotlib animation update function that visualizes:
    - Faint network background
    - Vibrant agent color
    - Agent trail flashing
    - Highlighted active link used in the current step
    """
    # Consistent, vibrant colors using tab10
    color_map = cm.get_cmap('tab10', len(agent_ids))
    agent_color_dict = {aid: color_map(i) for i, aid in enumerate(agent_ids)}

    # Track each agent's previous node to highlight the just-used link
    agent_last_positions = {aid: None for aid in agent_ids}

    def update(frame):
        ax.clear()
        G = snapshots[frame]['G']
        agents = snapshots[frame]['agents']

        # === Draw network with transparency ===
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150, node_color='lightgray', alpha=0.4)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.0, alpha=0.3)

        # === Draw current travel time as edge labels ===
        edge_labels = {
            (u, v): f"{round(attr.get('current_travel_time', 0), 1)}"
            for u, v, attr in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

        # === Draw agents and effects ===
        for agent in agents:
            aid = agent['agent_id']
            if aid not in agent_ids:
                continue

            color = agent_color_dict[aid]
            curr_node = agent['current_position']

            # === Highlight active link (just-used one from last node) ===
            prev_node = agent_last_positions.get(aid)
            if prev_node is not None and G.has_edge(prev_node, curr_node):
                a = 1
                # ax.plot(
                #     [pos[prev_node][0], pos[curr_node][0]],
                #     [pos[prev_node][1], pos[curr_node][1]],
                #     color=color, linewidth=4.0, alpha=0.8, zorder=1
                # )
            agent_last_positions[aid] = curr_node  # update for next frame

            # === Flash trail nodes ===
            # === Persistent comet-style trail (no fading) ===
            trail_nodes = set()
            for u, v, attr in G.edges(data=True):
                if attr.get('link_id') in agent.get('traveled_path', []):
                    trail_nodes.add(u)

            for node in trail_nodes:
                ax.scatter(*pos[node], s=90, color=color, edgecolors='none', zorder=2)

            for u, v, attr in G.edges(data=True):
                if attr.get('link_id') in agent.get('traveled_path', []):
                    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                            color=color, linewidth=1.0, alpha=0.6, zorder=1)

            # === Agent current position ===
            ax.scatter(*pos[curr_node], s=250, color=color, edgecolors='black', linewidths=1.2,
                       label=f"A{aid}", zorder=3)

        ax.set_title(f"Rollout Iteration {frame}", fontsize=14)
        ax.legend(loc='lower left', fontsize=8)
        ax.axis('off')

    return update
