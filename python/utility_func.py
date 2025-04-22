# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

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

    def compute_utility(self, net: Network, k_id, a_id):
        """
        net: reference to the full network (to look up attributes).
        k_id: ID of current link (the 'state')
        a_id: ID of next link (the 'action')

        Return the deterministic portion of utility v(a|k).
        This is typically sum of (beta_i * attribute_i).
        """
        link_obj = net.links[a_id]
        # example: link travel time
        time_ = link_obj.attributes.get('travel_time', 0.0)
        distance_ = link_obj.attributes.get('length', 0.0)
        
        # Suppose we do a simple linear combination:
        # v(a|k) = beta_time * time + beta_dist * distance
        v = 0.0
        if 'time' in self.beta:
            v += self.beta['time'] * time_
        if 'distance' in self.beta:
            v += self.beta['distance'] * distance_
        
        # If we needed turn-based attributes, we would also look up net.links[k_id].
        
        return v
