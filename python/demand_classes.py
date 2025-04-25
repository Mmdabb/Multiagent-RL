# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

class DemandRecord:
    def __init__(self, origin_link_id, destination_link_id, volume, origin_zone=None, destination_zone=None):
        self.origin_link_id = origin_link_id
        self.destination_link_id = destination_link_id
        self.volume = volume
        self.origin_zone = origin_zone
        self.destination_zone = destination_zone


class DemandSet:
    """
    Manages all demands. Typically you'd map (origin_link, destination_link) -> volume
    or (origin_zone, destination_zone) -> volume. Here we keep it simple by storing a list.
    """
    def __init__(self):
        self.demands = []  # list of DemandRecord
    
    def add_demand(self, origin_link, dest_link, volume, origin_zone=None, destination_zone=None):
        self.demands.append(DemandRecord(origin_link, dest_link, volume, origin_zone, destination_zone))

