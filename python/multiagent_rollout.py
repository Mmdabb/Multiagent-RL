import math
import random
from collections import defaultdict, Counter

class VehicleAgent:
    def __init__(self, agent_id, origin_link_id, destination_link_id):
        self.agent_id = agent_id
        self.origin = origin_link_id
        self.destination = destination_link_id

class MultiAgentPolicyIteration:
    def __init__(self, net, rl_model, vehicle_set):
        self.net = net
        self.rl_model = rl_model
        self.vehicles = vehicle_set
        self.current_policy = {}
        self.previous_policy = {}
        self.reward_history = []
        self.change_rate_history = []
        self.path_distribution_history = []

    def get_base_policy_path(self, origin, destination):
        dist = {lid: float('inf') for lid in self.net.links}
        prev = {}
        dist[origin] = 0
        queue = [(0, origin)]
        while queue:
            queue.sort()
            cost, u = queue.pop(0)
            if u == destination:
                break
            for succ_id in self.net.get_successor_links(u):
                travel_time = self.net.links[succ_id].attributes.get('travel_time', 1.0)
                alt = cost + travel_time
                if alt < dist[succ_id]:
                    dist[succ_id] = alt
                    prev[succ_id] = u
                    queue.append((alt, succ_id))
        path = []
        current = destination
        while current in prev:
            path.append(current)
            current = prev[current]
        if path:
            path.append(origin)
            path.reverse()
        return path if path and path[0] == origin else []

    def rollout_one_agent(self, agent, choice_probs, link_flows, max_steps=50):
        path = [agent.origin]
        current = agent.origin
        steps = 0
        while current != agent.destination and steps < max_steps:
            if current not in choice_probs:
                break
            probs = choice_probs[current]
            next_links = list(probs.keys())
            weights = []
            for a in next_links:
                u = self.rl_model.utility_func.compute_utility(self.net, current, a, link_flows)
                v = self.rl_model.get_value_function(agent.destination).get(a, -float('inf'))
                weights.append(math.exp((u + v) / self.rl_model.mu))
            if not weights:
                break
            total = sum(weights)
            weights = [w / total for w in weights]
            next_link = random.choices(next_links, weights=weights, k=1)[0]
            path.append(next_link)
            current = next_link
            steps += 1
        reward = -sum(self.net.links[link_id].attributes.get('travel_time', 1.0) for link_id in path)
        return path, reward

    def evaluate_policy_cost(self, policy):
        total_reward = 0.0
        for agent in self.vehicles:
            path = policy.get(agent.agent_id, [])
            reward = -sum(self.net.links[link_id].attributes.get('travel_time', 1.0) for link_id in path)
            total_reward += reward
        return total_reward / len(self.vehicles)

    def policy_iteration(self, n_iters=10, patience=3):
        for it in range(n_iters):
            print(f"\nPolicy Iteration Round {it+1}")
            all_choice_probs = {}
            for agent in self.vehicles:
                if agent.destination not in all_choice_probs:
                    all_choice_probs[agent.destination] = self.rl_model.get_choice_probabilities(agent.destination)

            new_policy = {}
            path_freq = Counter()
            total_reward = 0.0
            changed = 0
            link_flows = defaultdict(float)

            for agent in self.vehicles:
                probs = all_choice_probs[agent.destination]
                path, reward = self.rollout_one_agent(agent, probs, link_flows)
                new_policy[agent.agent_id] = path
                total_reward += reward
                for lid in path:
                    link_flows[lid] += 1.0
                path_str = '->'.join(path)
                path_freq[path_str] += 1

            if self.current_policy:
                for aid in new_policy:
                    old = self.current_policy.get(aid, [])
                    new = new_policy[aid]
                    if old != new:
                        changed += 1
                total = len(new_policy)
                print(f"  Changed paths: {changed} / {total} ({changed / total:.1%})")
            else:
                print("  First policy initialized.")
                changed = len(new_policy)

            avg_reward = total_reward / len(self.vehicles)
            print(f"  Avg reward this round: {avg_reward:.2f}")

            # Reward-based acceptance
            old_avg_reward = self.evaluate_policy_cost(self.current_policy) if self.current_policy else float('-inf')

            if avg_reward >= old_avg_reward:
                self.previous_policy = self.current_policy
                self.current_policy = new_policy
                self.reward_history.append(avg_reward)
                self.change_rate_history.append(changed / len(self.vehicles))
                self.path_distribution_history.append(path_freq)
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            # Early stopping condition
            if no_improve_rounds >= patience:
                print(f"\n️Early stopping triggered: no improvement for {patience} consecutive rounds.")
                break

    def get_link_flows(self):
        link_flows = defaultdict(float)
        for path in self.current_policy.values():
            for link_id in path:
                link_flows[link_id] += 1.0
        return link_flows