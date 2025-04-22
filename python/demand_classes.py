# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

class DemandRecord:
    def __init__(self, origin_link_id, destination_link_id, volume):
        self.origin_link_id = origin_link_id
        self.destination_link_id = destination_link_id
        self.volume = volume

class DemandSet:
    """
    Manages all demands. Typically you'd map (origin_link, destination_link) -> volume
    or (origin_zone, destination_zone) -> volume. Here we keep it simple by storing a list.
    """
    def __init__(self):
        self.demands = []  # list of DemandRecord
    
    def add_demand(self, origin_link, dest_link, volume):
        self.demands.append(DemandRecord(origin_link, dest_link, volume))
