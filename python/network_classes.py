# -*- coding: utf-8 -*-
"""

@author: mabbas10
"""

import csv
import math
from collections import defaultdict, deque

class Node:
    def __init__(self, node_id, zone_id=None, geometry=None):
        self.node_id = node_id
        self.zone_id = zone_id
        self.geometry = geometry  # store WKT or shapely geometry here

class Link:
    def __init__(self, link_id, start_node, end_node, attributes=None):
        """
        link_id : unique identifier for this link
        start_node: Node object or node_id
        end_node: Node object or node_id
        attributes: dict or custom structure for link properties (distance, cost, etc.)
        """
        self.link_id = link_id
        self.start_node = start_node
        self.end_node = end_node
        self.attributes = attributes if attributes else {}

class Network:
    """
    Manages the graph structure, adjacency, etc.
    """
    def __init__(self):
        self.nodes = {}
        self.links = {}
        # adjacency lists
        self.outgoing_links = defaultdict(list)  # node_id -> list of link_ids
        self.incoming_links = defaultdict(list)  # node_id -> list of link_ids

        # For time-expanded networks or DAG, store a topological order if needed
        self.topological_order = []

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def add_link(self, link: Link):
        self.links[link.link_id] = link
        start_id = link.start_node
        end_id = link.end_node
        self.outgoing_links[start_id].append(link.link_id)
        self.incoming_links[end_id].append(link.link_id)

    def build_topological_order(self):
        """
        Compute a topological ordering of the DAG using Kahn's algorithm.
        This is valid only if the graph is acyclic. 
        """
        # in-degree array
        in_degree = defaultdict(int)
        for node_id in self.nodes:
            in_degree[node_id] = 0
        
        for link_id, link in self.links.items():
            in_degree[link.end_node] += 1
        
        # queue for nodes of in-degree 0
        queue = deque()
        for node_id, deg in in_degree.items():
            if deg == 0:
                queue.append(node_id)

        topo_order = []
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            # decrement in-degree of successors
            for out_link_id in self.outgoing_links[current]:
                end_n = self.links[out_link_id].end_node
                in_degree[end_n] -= 1
                if in_degree[end_n] == 0:
                    queue.append(end_n)

        self.topological_order = topo_order

    def get_successor_links(self, link_id):
        """
        For a link L from node i->j, the successors are the links that start at j.
        """
        end_node = self.links[link_id].end_node
        return self.outgoing_links[end_node]

    def get_predecessor_links(self, link_id):
        """
        For a link L from i->j, the predecessors are the links that end at i.
        """
        start_node = self.links[link_id].start_node
        return self.incoming_links[start_node]

    # Potential helper for building a 'dummy' destination link, etc.
