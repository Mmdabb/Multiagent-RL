import numpy as np
from utils import softmax, trace_greedy_path_from_value_function, update_link_cost_bpr


def run_one_step_multiagent_rollout(G, agents, value_function_dict, mu=0.1, greedy=False, random_seed=None):
    """
    Perform one-step rollout per agent. If agent deviates from current plan, replan from current node.
    - Updates flows accordingly.
    - Updates BPR travel times.
    - Returns number of agents that completed their trip.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    completed_count = 0
    agent_indices = np.random.permutation(len(agents))

    for idx in agent_indices:
        agent = agents[idx]
        curr = agent['current_position']
        dest = agent['destination_node']


        V = value_function_dict[dest]

        if curr == dest:
            completed_count += 1
            continue

        successors = list(G.successors(curr))
        if not successors:
            print(f"Warning: Agent {agent['agent_id']} stuck at node {curr}. No successors.")
            break

        # Compute scores for each successor
        scores = [-G[curr][s]['current_travel_time'] + V.get(s, float('-inf')) for s in successors]

        if greedy:
            selected_idx = int(np.argmax(scores))
        else:
            probs = softmax(np.array(scores) / mu)
            selected_idx = np.random.choice(len(successors), p=probs)

        next_node = successors[selected_idx]
        next_link = (curr, next_node)

        # Move the agent
        agent['traveled_path'].append(G[curr][next_node]['link_id'])
        agent['current_position'] = next_node

        if idx == 226:
            print(f"agent {idx} planned links {agent['planned_links']}")


        # Replanning check: is this the same as planned?
        if not agent['planned_links'] or agent['planned_links'][0] != next_link:
            G[next_link[0]][next_link[1]]['flow'] += 1
            update_link_cost_bpr(G, next_link[0], next_link[1])

            if idx == 226:
                print(f"agent {idx} needs replan since next planned link {agent['planned_links'][0]} is not the same as the next selected link {next_link}")

            # Remove flow along old remaining path
            for link in agent['planned_links']:
                G[link[0]][link[1]]['flow'] -= 1
                # update_link_cost_bpr(G, link[0], link[1])
                # G[link[0]][link[1]]['flow'] = max(G[link[0]][link[1]]['flow'] - 1, 0)


            # Recompute new path from current node using V
            new_path = trace_greedy_path_from_value_function(G, next_node, dest, V)
            # print(f"agent {idx} needed replan and the new plan is {[next_link] + new_path}")

            agent['planned_links'] = new_path

            if idx == 226:
                print(f"agent {idx} new planned links {agent['planned_links']}")
                print(f"agent {idx} travelled links {agent['traveled_path']}")


            for link in agent['planned_links']:
                G[link[0]][link[1]]['flow'] += 1
                update_link_cost_bpr(G, link[0], link[1])
        else:
            # Continue down planned path, just remove used link
            agent['planned_links'] = agent['planned_links'][1:]


        # Update travel time via BPR
        # update_link_cost_bpr(G, curr, next_node)

    return completed_count
