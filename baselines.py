import networkx as nx 
import numpy as np
import itertools 
import community
import sklearn
import hdbscan


import msg_passing
import utils

def reweight_edges_louvain(g, nodes):
    avg_g = msg_passing.avg_edge_weights(g)
    avg_g = avg_g.subgraph(nodes).copy()
    for u, v in itertools.combinations(nodes, 2):
        if(u in avg_g.neighbors(v)):
            avg_g[u][v][0]["weight"] += 1 
        else:
            avg_g.add_edge(u, v, weight=1)
    return avg_g

def reweight_edges_hdbscan(g, nodes):
    avg_g = msg_passing.avg_edge_weights(g) 
    avg_g = avg_g.subgraph(nodes).copy() 
    for u, v in itertools.combinations(nodes, 2):
        if(u in avg_g.neighbors(v)):
            curr_weight = avg_g[u][v][0]["weight"] 
            if(curr_weight < 0):
                avg_g[u][v][0]["weight"] = 100
            else:
                avg_g[u][v][0]["weight"] = 2 - curr_weight
        else:
            avg_g.add_edge(u, v, weight=2)
    return avg_g

def set_node_index(g):
    for i, n in enumerate(g.nodes()):
        g.nodes()[n]['index'] = i

def cluster_louvain(g, nodes):
    ng = reweight_edges_louvain(g, nodes) 
    louvain_partition = community.best_partition(ng, weight='weight')
    node_labels = [louvain_partition[n] for n in nodes]
    return node_labels 

def cluster_hdbscan(g, nodes):
    distance_matrix = np.zeros((len(nodes), len(nodes)))
    ng = reweight_edges_hdbscan(g, nodes) 
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if(u == v):
                distance_matrix[i][j] = 0 
                continue
            distance_matrix[i][j] = ng[u][v][0]["weight"]
    model = hdbscan.HDBSCAN(metric='precomputed')
    model.fit(distance_matrix) 
    return model.labels_

node_ind_to_name = {}
dist_g = None
def reweight_distance(node1, node2):
    node1, node2 = node_ind_to_name[node1[0]], node_ind_to_name[node2[0]]
    global dist_g 
    if(node1 == node2):
        return 0
    dist = dist_g[node1][node2][0]["weight"]    
    return dist

def set_dist_g(g):
    global dist_g 
    dist_g = reweight_edges_hdbscan(g, g.nodes())

def evaluate_baseline_clustering(g, nodes, node_labels):
    # ignore -1 clusters
    labels_to_eval, nodes_to_eval = [], []
    for i in range(len(node_labels)):
        if(node_labels[i] >= 0):
            labels_to_eval.append(node_labels[i])
            nodes_to_eval.append(nodes[i])

    if(len(labels_to_eval) == 0):
        raise Exception("List of clusters to evaluate is empty")
    if(len(labels_to_eval) == 1):
        raise Exception("Single cluster provided")

    set_node_index(g)
    node_inds = [np.array([g.nodes()[n]['index']]) for n in nodes_to_eval]
    global node_ind_to_name
    for i, n in enumerate(nodes_to_eval):
        node_ind_to_name[node_inds[i][0]] = n
    set_dist_g(g)
    score = sklearn.metrics.silhouette_score(node_inds, labels_to_eval, metric=reweight_distance) 
    return score

def evaluate_issue_baseline(network_file, clustering_func, num_nodes=None, prune=True):
    g = msg_passing.load_graph_graphml(network_file) 
    if(prune):
        g, _ = msg_passing.prune_graph(g) 
    if(not num_nodes):
        num_nodes = len(g.nodes())
    nodes = utils.get_top_n_nodes(g, num_nodes)
    node_labels = clustering_func(g, nodes)
    score = evaluate_baseline_clustering(g, nodes, node_labels)
    return score, node_labels

def evaluate_multiple_issues_louvain(network_dir, network_names, network_suffix, num_nodes=None, prune=True):
    for name in network_names:
        filepath = network_dir + name + network_suffix + ".graphml"
        score, node_labels = evaluate_issue_baseline(filepath, cluster_louvain, num_nodes=num_nodes, prune=prune)

        num_clusters = len(set(node_labels))
        print(f"Louvain clustering results for {name} - num clusters: {num_clusters}, score: {score}")
        
def evaluate_multiple_issues_hdbscan(network_dir, network_names, network_suffix, num_nodes=None, prune=True):
    for name in network_names:
        filepath = network_dir + name + network_suffix + ".graphml"
        score, node_labels = evaluate_issue_baseline(filepath, cluster_hdbscan, num_nodes=num_nodes, prune=prune)

        num_clusters = len(set(node_labels))
        print(f"HDBSCAN clustering results for {name} - num clusters: {num_clusters}, score: {score}")