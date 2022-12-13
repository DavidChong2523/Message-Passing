import networkx as nx
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import pickle 
import os 
import random
import itertools

import utils
import parse_data

def load_graph_df(df, clean_data, source_col, target_col):
    g = nx.MultiGraph()
    df["publish_date"] = df["publish_date"].fillna("")
    df["url_host"] = df["url"].apply(parse_data.get_url_host)
    if(clean_data):
        parse_data.clean_df(df)
    

    edge_questions = defaultdict(set)
    for _, row in df.iterrows():
        edge_weight, ques = utils.get_ques_val(row["valence"])
        if(not edge_weight):
            continue

        # don't add duplicate questions to the graph
        source, target = row[source_col], row[target_col]
        if((ques in edge_questions[(source, target)]) or (ques in edge_questions[(target, source)])):
            continue
        edge_questions[(source, target)].add(ques)
        edge_questions[(target, source)].add(ques)
        
        source, target = row[source_col], row[target_col]
        g.add_edge(source, target,
            weight=edge_weight,
            valence=row["valence"],
            confidence=row["confidence"],
            publish_date=row["publish_date"],
            full_text=row["full_text"],
            ans_window=row["sub_window"],
            #summary=row["summary"],
            #keywords=row["keywords"],
            #publish_date=row["publish_date"],
            #authors=row["authors"],
            url=row["url"],
            #leaf_label=row["leaf_label"],
            #root_label=row["root_label"]
        )

    return g

def load_graph_csv(fname, clean_data=False, source_col="from_node", target_col="raw_answer"):
    df = pd.read_csv(fname)
    return load_graph_df(df, clean_data, source_col, target_col)

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
def initialize_node_values(g, size=1):
    for n in g.nodes():
        vec = utils.sample_spherical(1, ndim=size)
        vec = vec.reshape((size,))
        g.nodes()[n]["value"] = utils.unit_vec(vec)

def randomize_edge_weights(g, weight_list):
    for u, v, k in g.edges(keys=True):
        random_weight = random.choice(weight_list)
        g[u][v][k]["weight"] = random_weight 

def randomize_edges(g, weight_list):
    avg_g = avg_edge_weights(g) 
    random_g = nx.gnm_random_graph(len(avg_g.nodes()), len(avg_g.edges()), seed=1373849)
    random_g = utils.largest_connected_component(random_g)
    random_g = nx.MultiGraph(random_g)
    randomize_edge_weights(random_g, weight_list)
    node_labels = {}
    for i, n in enumerate(g.nodes()):
        node_labels[i] = n
    random_g = nx.relabel_nodes(random_g, node_labels)
    return random_g

def permute_edges(g):
    #adjacency_matrix = nx.adjacency_matrix(g, weight="weight") 
    adjacency_matrix = nx.to_numpy_matrix(g, weight="weight")
    permuted = np.random.permutation(adjacency_matrix) 
    for i in range(permuted.shape[0]):
        for j in range(permuted.shape[1]):
            if(i < j):
                permuted[i][j] = permuted[j][i]
    permuted_g = nx.from_numpy_matrix(permuted, create_using=nx.MultiGraph) 
    node_labels = {}
    for i, n in enumerate(g.nodes()):
        node_labels[i] = n
    permuted_g = nx.relabel_nodes(permuted_g, node_labels)
    return permuted_g 

def initialize_node_influence(g, n):
    for node in g.nodes():
        g.nodes()[node]["value"] = 0 
    for neighbor in g.neighbors(n):
        edge_weights = np.array([e["weight"] for e in g[n][neighbor].values()])
        avg_weight = np.average(edge_weights)
        g.nodes()[neighbor]["value"] = avg_weight 

# compute cosine distance loss over graph g
def loss_cos_dist(g):
    loss = [w*utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"])
            for u, v, w in g.edges(data="weight")]
    return sum(loss)
    
def avg_edge_weights(g):
    avg_g = nx.create_empty_copy(g)
    for u, v in g.edges():
        edge_weights = np.array([e["weight"] for e in g[u][v].values()])
        avg_weight = np.average(edge_weights)
        avg_g.add_edge(u, v, weight=avg_weight)
                
    return avg_g

def extreme_edge_weights(g):
    process_g = nx.create_empty_copy(g)
    for u, v in g.edges():
        edge_weights = [e["weight"] for e in g[u][v].values()]
        pos_weights = [e for e in edge_weights if e > 0]
        neg_weights = [e for e in edge_weights if e < 0]
        if(len(pos_weights) == 0 and len(neg_weights) == 0):
            continue
        elif(len(pos_weights) == 0):
            avg_weight = min(neg_weights) 
        elif(len(neg_weights) == 0):
            avg_weight = max(pos_weights)
        else: 
            avg_weight = np.average(np.array([min(neg_weights), max(pos_weights)]))
        process_g.add_edge(u, v, weight=avg_weight)
        
    return process_g
    

# iteratively remove auxiliary nodes from graph
# returns pruned graph, list of auxiliary nodes ordered such that m < n if dist(m, core node) < dist(n, core node)
def prune_graph(g):
    msg_g = g.copy()
    aux_nodes = []
    while(True):
        removed, num_removed = utils.remove_auxiliary_nodes(msg_g)
        aux_nodes += removed
        if(num_removed == 0):
            break
    aux_nodes = aux_nodes[::-1]
    return msg_g, aux_nodes

# populate auxiliary nodes based on core node values
# aux_nodes is an array of auxiliary nodes ordered such that m < n if dist(m, core node) < dist(n, core node)
# uninitialized nodes in g have a value equal to the zero vector
def set_auxiliary_values(g, aux_nodes):
    for n in aux_nodes:
        for n, neighbor, w in g.edges(n, data="weight"):
            # skip the uninitialized neighbor
            if(not np.any(g.nodes()[neighbor]["value"])):
                continue
            # assign to n based on initialized neighbor
            if(w > 0):
                g.nodes()[n]["value"] = g.nodes()[neighbor]["value"]
            elif(w < 0):
                # reflect neighbor value to opposite side of unit sphere
                g.nodes()[n]["value"] = -1*g.nodes()[neighbor]["value"]

# update target node with the gradient of the cos distance loss
# vals is a list of (weight, message node value)
def update_node_value(node_val, vals, eta_p, eta_n):
    next_val = np.copy(node_val)
    
    # HANDLE GRAD 0 AT LOCAL MAX?
    pos_nodes, pos_edges, neg_nodes, neg_edges = [], [], [], []
    for e, v in vals:
        if(e > 0):
            pos_nodes.append(v)
            pos_edges.append(e)
        elif(e < 0):
            neg_nodes.append(v)
            neg_edges.append(e)
    pos_nodes, pos_edges = np.array(pos_nodes), np.array(pos_edges) / np.sum(pos_edges)
    neg_nodes, neg_edges = np.array(neg_nodes), np.array(neg_edges) / np.sum(neg_edges)

    # DCHONG TESTING:
    #pos_weight_mag = sum([abs(p) for p in pos_edges])
    #neg_weight_mag = sum([abs(n) for n in neg_edges]) 
    #pos_weight = pos_weight_mag / (pos_weight_mag + neg_weight_mag)
    #neg_weight = neg_weight_mag / (pos_weight_mag + neg_weight_mag)

    if(len(pos_edges) > 0):
        pos_avg = utils.avg_vec(pos_nodes, weights=pos_edges)
        next_val -= eta_p * utils.vec_grad(node_val, pos_avg) # * pos_weight 
    if(len(neg_edges) > 0):
        neg_avg = utils.avg_vec(neg_nodes, weights=neg_edges)
        next_val += eta_n * utils.vec_grad(node_val, neg_avg) # * neg_weight 
    next_val = utils.unit_vec(next_val)
    return next_val    

# vals is a list of (weight, message node value)
def update_node_value_correctly_weighted(node_val, vals, eta_p, eta_n):
    next_val = np.copy(node_val)
    weights, nodes = [], []
    for e, v in vals:
        weights.append(e) 
        nodes.append(v)
        
    nodes, weights = np.array(nodes), np.array(weights / sum([abs(w) for w in weights]))
    for w, n in zip(weights, nodes):
        next_val -= (eta_p + eta_n)/2 * w * utils.vec_grad(node_val, n) 
    next_val = utils.unit_vec(next_val)
    return next_val    


# update node via message passing with neighbors
def update_node_message_passing(g, n, eta_p, eta_n):
    edge_vals = [(w, g.nodes[v]["value"]) for _, v, w in g.edges(n, data="weight")]    
    return update_node_value(g.nodes()[n]["value"], edge_vals, eta_p, eta_n)

# update node via message passing with random walks
# TODO: try degree penalty: sqrt(neighbor_deg/max_deg)
def update_node_random_walk(g, n, eta_p, eta_n, discount, path_length, batch_size, allow_loops=True):
    # stores tuples of (issue_weight, issue_vector)
    update_nodes = []
    for _ in range(batch_size):
        # initialize to 1/discount so the discount only applies on the second step of the path
        issue_weight = 1/discount
        curr_node = n 
        visited_nodes = set()
        for _ in range(path_length):
            next_node = random.choice(list(g.neighbors(curr_node)))
            if((not allow_loops) and (next_node in visited_nodes)):
                break
            issue_weight *= discount*g[curr_node][next_node][0]["weight"]
            issue_vec = g.nodes()[next_node]["value"]
            update_nodes.append((issue_weight, issue_vec))

            visited_nodes.add(next_node)
            curr_node = next_node

    next_val = update_node_value(g.nodes()[n]["value"], update_nodes, eta_p, eta_n)
    return next_val
        
def update_node_random_walk_correctly_weighted(g, n, eta_p, eta_n, discount, path_length, batch_size, allow_loops=True):
    # stores tuples of (issue_weight, issue_vector)
    update_nodes = []
    for _ in range(batch_size):
        # initialize to 1/discount so the discount only applies on the second step of the path
        issue_weight = 1/discount
        curr_node = n 
        visited_nodes = set()
        for _ in range(path_length):
            next_node = random.choice(list(g.neighbors(curr_node)))
            if((not allow_loops) and (next_node in visited_nodes)):
                break
            issue_weight *= discount*g[curr_node][next_node][0]["weight"]
            issue_vec = g.nodes()[next_node]["value"]
            update_nodes.append((issue_weight, issue_vec))

            visited_nodes.add(next_node)
            curr_node = next_node

    next_val = update_node_value_correctly_weighted(g.nodes()[n]["value"], update_nodes, eta_p, eta_n)
    return next_val

def update_node_random_walk_degree_penalty(g, n, eta_p, eta_n, discount, path_length, batch_size, allow_loops=True):
    max_degree = utils.node_degrees(g)[0][1]

    # stores tuples of (issue_weight, issue_vector)
    update_nodes = []
    for _ in range(batch_size):
        # initialize to 1/discount so the discount only applies on the second step of the path
        issue_weight = 1/discount
        curr_node = n 
        visited_nodes = set()
        for _ in range(path_length):
            next_node = random.choice(list(g.neighbors(curr_node)))
            if((not allow_loops) and (next_node in visited_nodes)):
                break
            degree_weight = np.sqrt(g.degree(next_node) / max_degree)
            issue_weight *= discount*degree_weight*g[curr_node][next_node][0]["weight"]
            issue_vec = g.nodes()[next_node]["value"]
            update_nodes.append((issue_weight, issue_vec))

            visited_nodes.add(next_node)
            curr_node = next_node

    next_val = update_node_value(g.nodes()[n]["value"], update_nodes, eta_p, eta_n)
    return next_val

def update_node_random_walk_degree_penalty_correctly_weighted_update(g, n, eta_p, eta_n, discount, path_length, batch_size, allow_loops=True):
    max_degree = utils.node_degrees(g)[0][1]

    # stores tuples of (issue_weight, issue_vector)
    update_nodes = []
    for _ in range(batch_size):
        # initialize to 1/discount so the discount only applies on the second step of the path
        issue_weight = 1/discount
        curr_node = n 
        visited_nodes = set()
        for _ in range(path_length):
            next_node = random.choice(list(g.neighbors(curr_node)))
            if((not allow_loops) and (next_node in visited_nodes)):
                break
            degree_weight = np.sqrt(g.degree(next_node) / max_degree)
            issue_weight *= discount*degree_weight*g[curr_node][next_node][0]["weight"]
            issue_vec = g.nodes()[next_node]["value"]
            update_nodes.append((issue_weight, issue_vec))

            visited_nodes.add(next_node)
            curr_node = next_node

    next_val = update_node_value_correctly_weighted(g.nodes()[n]["value"], update_nodes, eta_p, eta_n)
    return next_val

def pass_messages_on_graph(g, update_node_func, eta_p, eta_n, iters, use_heat, print_period=None, save_period=None, history={}, **kwargs):
    # initialize history
    diagnostic_hist = defaultdict(list)
    if(save_period):
        diagnostic_hist["SAVE_PER"] = [save_period]
    for k in history.keys():
        history[k].append(g.nodes()[k]["value"])
        
    heat = 0
    for iter in range(iters): 
        if(use_heat):
            heat = 1 - (iter+1)/(iters)

        for n in g.nodes():
            if(np.random.random() < heat): 
                # simulated annealing
                random_vec = utils.random_dir(g.nodes()[n]["value"].shape)
                next_val = g.nodes()[n]["value"] + (eta_p+eta_n)/2 * random_vec
                next_val = utils.unit_vec(next_val)
            else:
                next_val = update_node_func(g, n, eta_p, eta_n, **kwargs)
            g.nodes()[n]["next_value"] = next_val
            
        # diagnostic info for change in node values  
        save_cond = save_period and (iter % save_period == 0 or iter == iters-1)
        print_cond = print_period and (iter % print_period == 0 or iter == iters-1)
        if(save_cond or print_cond):
            update_mag = sum([np.linalg.norm(g.nodes()[n]["next_value"] - g.nodes()[n]["value"]) for n in g.nodes()])
            loss = loss_cos_dist(g)          
        if(save_cond):
            diagnostic_hist["UPDATE_MAG"].append(update_mag)
            diagnostic_hist["LOSS"].append(loss)
            for k in history.keys():
                history[k].append(g.nodes()[k]["next_value"])
        if(print_cond):       
            print("iteration " + str(iter) + ": update mag:", update_mag, "loss:", loss)
 
        for n in g.nodes():
            g.nodes()[n]["value"] = g.nodes()[n]["next_value"]

    return history, diagnostic_hist

def pass_messages(g, eta_p, eta_n, iters, use_heat, pruning=True, print_period=None, save_period=None, history={}):
    msg_g, aux_nodes = g, []
    if(pruning):
        msg_g, aux_nodes = prune_graph(g)

    history, diagnostic_hist = pass_messages_on_graph(
        msg_g, update_node_message_passing, eta_p, eta_n, iters, use_heat, 
        print_period=print_period, save_period=save_period, history=history
    )

    for n in msg_g.nodes():
        g.nodes()[n]["value"] = msg_g.nodes()[n]["value"]
    if(pruning):
        set_auxiliary_values(g, aux_nodes)

    return history, diagnostic_hist

def pass_messages_with_random_walks(g, eta_p, eta_n, iters, use_heat, pruning=True, print_period=None, save_period=None, history={}, discount=1, path_length=10, batch_size=10, update_func=update_node_random_walk):
    msg_g, aux_nodes = g, []
    if(pruning):
        msg_g, aux_nodes = prune_graph(g)

    msg_g = avg_edge_weights(msg_g)
    #msg_g = extreme_edge_weights(msg_g)

    history, diagnostic_hist = pass_messages_on_graph(
        msg_g, update_func, eta_p, eta_n, iters, use_heat, 
        print_period=print_period, save_period=save_period, history=history, 
        discount=discount, path_length=path_length, batch_size=batch_size
    )

    for n in msg_g.nodes():
        g.nodes()[n]["value"] = msg_g.nodes()[n]["value"]
    if(pruning):
        set_auxiliary_values(g, aux_nodes)

    return history, diagnostic_hist

# input is initialized larget connected component of graph
# history is a dictionary of {date: {node_name: []}}
# save_path is a directory/filename_prefix, "{DATE}.graphml" will be added on
def pass_messages_time_series(dates, g, eta_p, eta_n, iters, use_heat, stop_thresh=None, print_period=None, save_period=None, history={}, remove_undated_edges=False, save_path=None):
    g = g.copy()
    if(remove_undated_edges):
        to_remove = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d["publish_date"] == ""]
        g.remove_edges_from(to_remove)
        
    results = []
    for i, date in enumerate(dates):
        print(f"Training on dates before {date}, graph {i+1}/{len(dates)}")
        curr_g = g.copy()
        to_remove = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if d["publish_date"] > date]
        curr_g.remove_edges_from(to_remove)
        curr_hist = history.get(date, {})
        node_hist, diagnostic_hist = pass_messages(curr_g, eta_p, eta_n, iters, use_heat, stop_thresh=stop_thresh, print_period=print_period, save_period=save_period, history=curr_hist)
        results.append((curr_g, node_hist, diagnostic_hist))

        if(save_path):
            full_path = save_path + "_" + date.replace(" ", "_").replace(":", "-")
            full_path_graph = full_path + ".graphml"
            full_path_hist = full_path + ".pkl"
            save_graph(curr_g, full_path_graph)
            save_history(node_hist, diagnostic_hist, full_path_hist)

    return results

def generate_graph_date_range(g, step):
    DATE_FORMAT = "YYYY-MM-DD"
    dates = [d["publish_date"] for _, _, d in g.edges(data=True) if d["publish_date"] != ""]
    dates = sorted(list(set(dates)))
    start, end = str(dates[0])[:len(DATE_FORMAT)+1], str(dates[-1])[:len(DATE_FORMAT)+1]
    return utils.generate_date_range(start, end, step)


