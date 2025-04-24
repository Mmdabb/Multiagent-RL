# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

import math
from collections import defaultdict, deque
from network_classes import Network
from utility_func import UtilityFunction

class RecursiveLogitModel:
    """
    Computes the RL value function V_d(link) for a given destination link 'd'
    and then yields link-choice probabilities P_d(a|k).
    Assumes i.i.d. Gumbel errors with scale mu.
    """
    def __init__(self, net: Network, utility_func: UtilityFunction, mu=1.0):
        self.net = net
        self.utility_func = utility_func
        self.mu = mu

        # Store the value function and choice probabilities, once computed
        # For each destination link 'd', we might store an array V_d[link_id].
        self.value_cache = {}
        self.prob_cache = {}

    def compute_value_function(self, dest_id):
        """
        For an absorbing 'destination link' dest_id, compute V_dest(link_id) for all links.

        Because we have a DAG, we can solve in reverse topological order of links.
        We'll store results in a dict: Vdest[link_id] = ...
        """
        # For convenience, let's define a local dictionary:
        Vdest = defaultdict(float)

        # 1) Mark the absorbing link's value as 0
        Vdest[dest_id] = 0.0

        # We'll need an ordering of links that ensures we process successors first.
        # Easiest is to define our own link-level topological order:
        link_topo = self.build_link_topo_sort()

        # We'll walk link_topo in reverse
        for lk in reversed(link_topo):
            if lk == dest_id:
                continue
            successors = self.net.get_successor_links(lk)
            if len(successors) == 0:
                # If no successor links, treat as a dead-end (unless it's the absorbing link)
                # You might set Vdest[lk] = -999999 or something to reflect no valid path
                Vdest[lk] = float('-inf')
            else:
                # V(k) = mu * log( sum_{a in successors} exp( [v(a|k) + V(a)] / mu ) )
                # Implementation detail: typically we factor out 1/mu
                sum_exp = 0.0
                for a in successors:
                    v_ak = self.utility_func.compute_utility(self.net, lk, a)
                    # exponent = (v_ak + Vdest[a]) / mu
                    exponent = (v_ak + Vdest[a]) / self.mu
                    sum_exp += math.exp(exponent)

                if sum_exp <= 1e-300: 
                    # to avoid log(0) issues
                    Vdest[lk] = float('-inf')
                else:
                    Vdest[lk] = self.mu * math.log(sum_exp)

        return Vdest

    def build_link_topo_sort(self):
        """
        Construct a topological ordering of links themselves.
        Each link is considered a 'node' in the meta-graph, 
        and there's a directed edge from link k->a if a starts where k ends.
        We'll do a standard DAG topological sort of links.
        """
        # step 1: in-degree of each link
        in_degree_links = defaultdict(int)
        for link_id in self.net.links:
            in_degree_links[link_id] = 0

        # For each link, find its successors and increment their in-degree
        for k in self.net.links:
            succs = self.net.get_successor_links(k)
            for a in succs:
                in_degree_links[a] += 1

        # step 2: Kahn's algorithm
        queue = deque()
        for lk, deg in in_degree_links.items():
            if deg == 0:
                queue.append(lk)

        topo_order = []
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            for nxt in self.net.get_successor_links(current):
                in_degree_links[nxt] -= 1
                if in_degree_links[nxt] == 0:
                    queue.append(nxt)

        return topo_order

    def get_value_function(self, dest_id):
        """
        Public method to retrieve the value function for a given destination link.
        Caches the result to avoid recomputing if repeated calls.
        """
        if dest_id not in self.value_cache:
            Vd = self.compute_value_function(dest_id)
            self.value_cache[dest_id] = Vd
        return self.value_cache[dest_id]

    def link_choice_probability(self, k, a, dest_id):
        """
        P_d(a|k) = exp( (v(a|k) + V_d(a)) / mu ) / sum_{a' in A(k)} ...
        We'll compute it on-the-fly. 
        For repeated queries, you might want to cache these probabilities as well.
        """
        Vd = self.get_value_function(dest_id)  # get or compute
        successors = self.net.get_successor_links(k)
        denom = 0.0
        for a_prime in successors:
            v_ap = self.utility_func.compute_utility(self.net, k, a_prime)
            exponent = (v_ap + Vd[a_prime]) / self.mu
            denom += math.exp(exponent)

        v_ak = self.utility_func.compute_utility(self.net, k, a)
        numerator = math.exp((v_ak + Vd[a]) / self.mu)

        if denom < 1e-300:
            return 0.0
        else:
            return numerator / denom

    def get_choice_probabilities(self, dest_id):
        """
        Return a dictionary prob[k][a] = P_d(a|k).
        """
        Vd = self.get_value_function(dest_id)
        prob = defaultdict(dict)
        for k in self.net.links:
            succs = self.net.get_successor_links(k)
            sum_exp = 0.0
            exps = {}
            for a in succs:
                val = self.utility_func.compute_utility(self.net, k, a) + Vd[a]
                exps[a] = math.exp(val / self.mu)
                sum_exp += exps[a]
            for a in succs:
                if sum_exp < 1e-300:
                    prob[k][a] = 0.0
                else:
                    prob[k][a] = exps[a] / sum_exp
        return prob
