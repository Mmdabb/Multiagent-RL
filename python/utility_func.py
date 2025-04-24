# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

from network_classes import Network
class UtilityFunction:
    """
    Encapsulates how to compute the deterministic part of utility for choosing link a
    from the current link k.
    """
    def __init__(self, beta):
        """
        beta: a dict or list of parameters, e.g. { 'time': -0.01, 'distance': -0.05, ... }
        or a direct numpy array.
        """
        self.beta = beta

    def compute_utility(self, net: Network, k_id, a_id, link_flows=None):
        """
        net: reference to the full network (to look up attributes).
        k_id: ID of current link (the 'state')
        a_id: ID of next link (the 'action')

        Return the deterministic portion of utility v(a|k).
        This is typically sum of (beta_i * attribute_i).
        """
        link_obj = net.links[a_id]
        # example: link travel time
        base_time = link_obj.attributes.get('travel_time', 0.0)
        distance_ = link_obj.attributes.get('length', 0.0)

        # Mean Field: adjust travel time based on flow
        if link_flows is not None:
            flow = link_flows.get(a_id, 0.0)
            time_ = base_time * (1 + 0.15 * (flow / 100.0))  # BPR-like adjustment
        else:
            time_ = base_time
        
        # Suppose we do a simple linear combination:
        # v(a|k) = beta_time * time + beta_dist * distance
        v = 0.0
        if 'time' in self.beta:
            v += self.beta['time'] * time_
        if 'distance' in self.beta:
            v += self.beta['distance'] * distance_
        
        # If we needed turn-based attributes, we would also look up net.links[k_id].
        
        return v
