# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:14:27 2025

@author: mabbas10
"""
from network_classes import Node, Link, Network
from demand_classes import DemandSet
from utility_func import UtilityFunction
from recursive_logit import RecursiveLogitModel
from rl_static_assignment import RLStaticAssigner
from multiagent_rollout import MultiAgentPolicyIteration, VehicleAgent

def main():
    # 1) Build a small DAG network in code (or parse from CSV)
    net = Network()
    # Create some nodes
    for i in range(1, 6):
        net.add_node(Node(node_id=i))

    # Suppose link_id=some integer or string, from->to
    # Let's build a small DAG:
    net.add_link(Link(link_id='L1', start_node=1, end_node=2, 
                      attributes={'length':2.0, 'travel_time':1.0}))
    net.add_link(Link(link_id='L2', start_node=2, end_node=3,
                      attributes={'length':3.0, 'travel_time':2.0}))
    net.add_link(Link(link_id='L3', start_node=2, end_node=4,
                      attributes={'length':1.0, 'travel_time':0.5}))
    net.add_link(Link(link_id='L4', start_node=4, end_node=5,
                      attributes={'length':2.0, 'travel_time':2.0}))
    net.add_link(Link(link_id='L5', start_node=3, end_node=5,
                      attributes={'length':1.0, 'travel_time':1.0}))

    # We treat node 5 as a pseudo-destination that might have a dummy link
    # but for simplicity let's say link_id='DestLink' goes from 5->5. (Absorbing.)
    # Or we can just treat link 5 as absorbing if it has no successors.
    # If you prefer a special dummy link, do that:
    net.add_link(Link(link_id='D', start_node=5, end_node=5,
                      attributes={'length':0.0, 'travel_time':0.0}))

    # We don't strictly need topological order at the node level for the RL model,
    # but let's do it if we want it for something else:
    net.build_topological_order()

    # 2) Build a demand set
    demands = DemandSet()
    # Suppose we say origin is L1 (which starts at node 1->2) for an OD volume of 100
    # and the "destination link" is 'D' (the absorbing link).
    # In many designs, you'd pick a special link from node=1->1 as the origin, but let's keep it simple:
    demands.add_demand('L1', 'D', 100.0)

    # 3) Build the utility function & RL model
    beta_params = {'time': -0.5, 'distance': -0.2}  # negative means we prefer less time/distance
    util_func = UtilityFunction(beta_params)
    rl_model = RecursiveLogitModel(net, util_func, mu=1.0)

    # 4) Run the assignment
    assigner = RLStaticAssigner(net, rl_model)
    assigned_flows = assigner.assign(demands)

    # 5) Print results
    for link_id, flow_val in assigned_flows.items():
        print(f"Link {link_id}: Flow={flow_val:.3f}")

if __name__ == "__main__":
    main()
