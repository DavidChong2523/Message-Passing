import networkx as nx
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import pickle 

import utils

def load_graph_csv(fname):
    g = nx.MultiGraph()
    df = pd.read_csv(fname)
    for _, row in df.iterrows():
        edge_weight = utils.get_ques_val(row["valence"])
        if(not edge_weight):
            continue

        source, target = row["source"].lower(), row["target"].lower()
        g.add_edge(source, target,
                   weight=edge_weight,
                   valence=row["valence"],
                   confidence=row["confidence"],
                   full_text=row["full_text"],
                   #summary=row["summary"],
                   #keywords=row["keywords"],
                   #publish_date=row["publish_date"],
                   #authors=row["authors"],
                   #url=row["url"],
                   #leaf_label=row["leaf_label"],
                   #root_label=row["root_label"]
                  )
    return g

def load_graph_graphml(fname):
    g = nx.read_graphml(fname)
    g = nx.MultiGraph(g)
    attrs = nx.get_node_attributes(g, "value")
    for k, v in attrs.items():
        attrs[k] = np.array(json.loads(v))
    nx.set_node_attributes(g, attrs, "value")
    return g

def save_graph(g, fname):
    def serialize_vals(g, val_name):
        attrs = nx.get_node_attributes(g, val_name)
        for k, v in attrs.items():
            attrs[k] = json.dumps(v.tolist())
        nx.set_node_attributes(g, attrs, val_name)

    sg = g.copy()
    serialize_vals(sg, "value")
    serialize_vals(sg, "next_value")

    sg_cyto = nx.MultiDiGraph(sg)
    nx.write_graphml(sg, fname)
    nx.write_graphml(sg_cyto, fname[:-1*len(".graphml")]+"_cyto.graphml")

def save_history(hist, diagnostic_hist, save_file):
    save_obj = {"hist": hist, "diagnostic_hist": diagnostic_hist}
    with open(save_file, "wb") as f:
        pickle.dump(save_obj, f)
        
def load_history(save_file):
    with open(save_file, "rb") as f:
        save_obj = pickle.load(f)
    return save_obj["hist"], save_obj["diagnostic_hist"]

# initialize node values as unit vector
def initialize_node_values(g, mean, std, size=1):
    for n in g.nodes():
        vec = np.random.normal(loc=mean, scale=std, size=size)
        g.nodes()[n]["value"] = utils.unit_vec(vec)

# update node n with the gradient of the cos distance loss
def update_cos_dist(g, n, eta_p, eta_n):
    # store tuples of (adjacent node value, edge weight)
    edge_vals = [(g.nodes[v]["value"], w) for _, v, w in g.edges(n, data="weight")] 
    next_val = np.copy(g.nodes()[n]["value"])
    
    # HANDLE GRAD 0 AT LOCAL MAX?
    pos_nodes, pos_edges, neg_nodes, neg_edges = [], [], [], []
    for v, e in edge_vals:
        if(e > 0):
            pos_nodes.append(v)
            pos_edges.append(e)
        elif(e < 0):
            neg_nodes.append(v)
            neg_edges.append(e)
    pos_nodes, pos_edges = np.array(pos_nodes), np.array(pos_edges) / np.sum(pos_edges)
    neg_nodes, neg_edges = np.array(neg_nodes), np.array(neg_edges) / np.sum(neg_edges)
    if(len(pos_edges) > 0):
        pos_avg = utils.avg_vec(pos_nodes, weights=pos_edges)
        next_val -= eta_p * utils.vec_grad(g.nodes()[n]["value"], pos_avg)     
    if(len(neg_edges) > 0):
        neg_avg = utils.avg_vec(neg_nodes, weights=neg_edges)
        next_val += eta_n * utils.vec_grad(g.nodes()[n]["value"], neg_avg) 
    next_val = utils.unit_vec(next_val)
    return next_val    

# compute cosine distance loss over graph g
def loss_cos_dist(g):
    loss = [w*utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"])
            for u, v, w in g.edges(data="weight")]
    return sum(loss)

def propagate_messages(g, eta_p, eta_n, heat=0):
    for n in g.nodes():
        #eta_p, eta_n = adjust_lr(n, g, eta_p, eta_n)
        
        # simulated annealing
        if(np.random.random() < heat):
            direction = np.random.random(size=g.nodes()[n]["value"].shape) - 0.5
            direction = utils.unit_vec(direction)
            next_val = g.nodes()[n]["value"] + (eta_p+eta_n)/2 * direction
            next_val = utils.unit_vec(next_val)
        else:
            next_val = update_cos_dist(g, n, eta_p, eta_n) 
    
        g.nodes()[n]["next_value"] = next_val
        
    # diagnostic info for change in node values
    update_mag = sum([np.linalg.norm(g.nodes()[n]["next_value"] - g.nodes()[n]["value"]) for n in g.nodes()])
    loss = loss_cos_dist(g) 
    
    # update node values
    for n in g.nodes():
        g.nodes()[n]["value"] = g.nodes()[n]["next_value"]
        
    return update_mag, loss    

def iterate(g, eta_p, eta_n, iters, stop_thresh=None, use_heat=False, print_period=None, history={}):
    # initialize history
    for k in history.keys():
        history[k].append(g.nodes()[k]["value"])
    diagnostic_hist = defaultdict(list)
        
    heat = 0
    i = 0
    while(True):
        if(use_heat):
            heat = 1 - (i+1)/(iters)
        update_mag, loss = propagate_messages(g, eta_p, eta_n, heat=heat)
        
        for k in history.keys():
            history[k].append(g.nodes()[k]["value"])
        
        diagnostic_hist["UPDATE_MAG"].append(update_mag)
        diagnostic_hist["LOSS"].append(loss)
        
        if(print_period and (i % print_period == 0 or i == iters-1)):
            print("iteration " + str(i) + ": update mag:", update_mag, "loss:", loss)
            
        i += 1
        if(i == iters or (stop_thresh and update_mag < stop_thresh)):
            break
        
    return history, diagnostic_hist
            