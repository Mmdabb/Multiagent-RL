# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

class RLStaticAssigner:
    """
    Given a RL model (which can compute choice probabilities) and a set of demands,
    perform the usual forward flow calculation in a DAG to get expected flows.
    """
    def __init__(self, net: Network, rl_model: RecursiveLogitModel):
        self.net = net
        self.rl_model = rl_model
        # store flows in link_flow[d][link_id] = float

    def assign(self, demands: DemandSet):
        """
        We assume each DemandRecord says: 'origin_link_id', 'destination_link_id', 'volume'.
        We'll compute F_d(k) for each link k, and each demand record's destination d,
        then sum up to get final flows.
        """
        assigned_flows = defaultdict(float)  # link -> total flow from all OD
        # process each demand record independently, then add them up.
        for drec in demands.demands:
            origin = drec.origin_link_id
            dest = drec.destination_link_id
            vol = drec.volume
            flows_for_this_OD = self.forward_flow(dest, origin, vol)
            # Add to the total assigned_flows
            for lk_id, fl in flows_for_this_OD.items():
                assigned_flows[lk_id] += fl

        return assigned_flows

    def forward_flow(self, dest_link_id, origin_link_id, demand_volume):
        """
        Solve the system:
           F_d(k) = G_d(k) + sum_{h in pred(k)} P_d(k|h)*F_d(h).
        But here we do it in a topological link ordering. 
        We'll store F_d(k) in a dictionary.

        We assume G_d(k) = demand_volume if k == origin_link_id, else 0.
        """
        # 1) Build link-level topological order if not already built:
        link_topo = self.rl_model.build_link_topo_sort()

        # 2) store F(k)
        Fd = defaultdict(float)
        Fd[origin_link_id] = demand_volume

        # 3) Process in topological order (i.e. start from the 'origins')
        #    Then each link can push flow to its successors.
        #    Because it's a DAG, once processed 'lk', there's no going back.
        for lk in link_topo:
            if Fd[lk] <= 1e-12:  # no flow to distribute from this link
                continue
            prob_sum = 0.0
            succs = self.net.get_successor_links(lk)
            if lk == dest_link_id:
                # It's the absorbing link. Do nothing more.
                continue
            for a in succs:
                p = self.rl_model.link_choice_probability(lk, a, dest_link_id)
                # push flow forward
                Fd[a] += Fd[lk] * p
        
        return Fd
