
import csv
from network_classes import Node, Link, Network
from demand_classes import DemandSet
from collections import defaultdict

def load_network_from_csv(node_file, link_file):
    net = Network()

    # Load nodes
    with open(node_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['node_id'])
            zone_id = int(row['zone_id']) if 'zone_id' in row and row['zone_id'] else None
            net.add_node(Node(node_id=node_id, zone_id=zone_id))

    # Load links
    with open(link_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_id = row['link_id']
            start = int(row['from_node_id'])
            end = int(row['to_node_id'])
            length = float(row['vdf_length_mi'])
            lanes = float(row['lanes'])
            cap_per_lane = float(row['capacity'])
            speed = float(row['vdf_free_speed_mph'])
            travel_time = (length / speed) * 60  # in minutes
            attributes = {
                'length': length,
                'travel_time': travel_time,
                'capacity': lanes * cap_per_lane
            }
            net.add_link(Link(link_id, start, end, attributes))

    return net

def load_od_demand(demand_file, net):
    demand_set = DemandSet()
    od_map = defaultdict(lambda: {'out': [], 'in': []})

    for link in net.links.values():
        if link.start_node in net.nodes and net.nodes[link.start_node].zone_id == link.start_node:
            od_map[link.start_node]['out'].append(link.link_id)
        if link.end_node in net.nodes and net.nodes[link.end_node].zone_id == link.end_node:
            od_map[link.end_node]['in'].append(link.link_id)

    with open(demand_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            oz = int(row['o_zone_id'])
            dz = int(row['d_zone_id'])
            vol = float(row['volume'])

            if od_map[oz]['out'] and od_map[dz]['in']:
                origin_link = od_map[oz]['out'][0]
                dest_link = od_map[dz]['in'][0]
                demand_set.add_demand(origin_link, dest_link, vol)

    return demand_set
