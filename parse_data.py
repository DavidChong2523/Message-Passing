import networkx as nx
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import pickle 
import os 
import random
import itertools
import sklearn.cluster
import sklearn.metrics
import urllib
import hdbscan 

import utils
import msg_passing


# remove leading 'the' from span
def remove_prefix_the(span):
    span = span.strip().lower()
    words = span.split(" ")
    while(len(words) > 0):
        if(words[0] == "the"):
            words = words[1:]
        else:
            break
    return " ".join(words)

# remove ending possessives: 's's's's or s'
def remove_possessive(span):
    span = span.strip().lower()
    words = span.split(" ")
    while(len(words) > 0):
        # possible apostrophe suffixes: "'s", "’s", "'"", "" 
        if(words[-1] == "'s" or words[-1] == "’s" or words[-1] == "'" or words[-1] == ""):
            words = words[:-1]
        elif(words[-1][-2:] == "'s" or words[-1][-2:] == "’s"):
            words[-1] = words[-1][:-2]
        elif(words[-1][-1] == "'"):
            words[-1] = words[-1][:-1]
        else:
            break
    return " ".join(words)

def clean_node_name(span):
    if(not isinstance(span, str)):
        # todo: use logging package
        print(f"[clean_node_name] span '{span}' is not type string")
        return "" 

    span = span.lower()
    span = remove_prefix_the(span) 
    span = remove_possessive(span) 
    return span

def clean_df(df, source_col="from_node", target_col="raw_answer"):
    for i, row in df.iterrows():
        source, target = row[source_col], row[target_col]
        processed_source, processed_target = clean_node_name(source), clean_node_name(target)
        df.loc[i, [source_col]] = [processed_source]
        df.loc[i, [target_col]] = [processed_target]

    # remove rows with empty source or target
    df[source_col].replace("", np.nan, inplace=True)
    df[target_col].replace("", np.nan, inplace=True) 
    df.dropna(subset=[source_col, target_col], inplace=True)

def get_url_host(url):
    return urllib.parse.urlparse(url).hostname

# input must be a connected graph
def compute_network_stats(g, include_diameter=True): 
    stats = {}
    stats["num_nodes"] = len(g.nodes())
    stats["num_edges"] = len(g.edges()) 
    stats["edges/node"] = stats["num_edges"] / stats["num_nodes"]

    weights = [w for _, _, w in g.edges(data="weight")] 
    stats["num_pos_edges"] = sum([1 for w in weights if w > 0])
    stats["num_neg_edges"] = sum([1 for w in weights if w < 0])
    stats["pos/neg edges"] = stats["num_pos_edges"] / stats["num_neg_edges"]
    if(include_diameter):
        stats["diameter"] = nx.diameter(g) 

    # DCHONG TESTING
    #weight_counts = [(d["weight"], d["count"]) for _, _, d in g.edges(data=True)]
    #stats["num_pos_edges_count_over_1"] = sum([1 for wc in weight_counts if wc[0] > 0 and wc[1] > 1])
    #stats["num_neg_edges_count_over_1"] = sum([1 for wc in weight_counts if wc[0] < 0 and wc[1] > 1])
    #stats["num_max_pos_edges_count_over_1"] = sum([1 for wc in weight_counts if wc[0] == 1 and wc[1] > 1])

    return stats


def generate_confusion_matrix(g, nodes):
    confusion_matrix = np.zeros((len(nodes), len(nodes)))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            confusion_matrix[i][j] = utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"])
    return confusion_matrix

def cluster_nodes_coclustering(g, nodes, num_clusters, random_state=100):
    confusion_matrix = generate_confusion_matrix(g, nodes)
    model = sklearn.cluster.SpectralCoclustering(n_clusters=num_clusters, random_state=random_state) 
    model.fit(confusion_matrix) 
    return model.row_labels_


def cluster_nodes_hdbscan(g, nodes):
    # cosine distance
    distance_matrix = generate_confusion_matrix(g, nodes)
    model = hdbscan.HDBSCAN(metric='precomputed')
    model.fit(distance_matrix) 
    return model.labels_

def generate_clustered_confusion_matrix(g, nodes, top_n_nodes=None, drop_noise=False):
    labels = cluster_nodes_hdbscan(g, nodes)

    processed_labels, processed_nodes = [], []
    if(drop_noise):
        for i in range(len(nodes)):
            if(labels[i] >= 0):
                processed_labels.append(labels[i])
                processed_nodes.append(nodes[i])
    labels = processed_labels
    nodes = processed_nodes

    nodes_to_display = []
    labels_to_display = []
    if(not top_n_nodes):
        top_n_nodes = len(g.nodes())
    for n in utils.get_top_n_nodes(g, len(g.nodes())):
        for i in range(len(nodes)):
            if(nodes[i] == n):
                nodes_to_display.append(n)
                labels_to_display.append(labels[i])
        if(len(nodes_to_display) == top_n_nodes):
            break

    row_inds = np.argsort(labels_to_display) 

    confusion_matrix = generate_confusion_matrix(g, nodes_to_display)
    confusion_matrix = confusion_matrix[row_inds] 
    confusion_matrix = confusion_matrix[:, row_inds] 
    display_row = [nodes_to_display[i] for i in row_inds]
    return display_row, confusion_matrix

def evaluate_clustering(g, nodes, labels):
    node_vals = [g.nodes()[n]["value"] for n in nodes]
    
    # ignore -1 clusters
    evaluate_labels, evaluate_node_vals = [], []
    for i in range(len(labels)):
        if(labels[i] >= 0):
            evaluate_labels.append(labels[i])
            evaluate_node_vals.append(node_vals[i])

    if(len(evaluate_labels) == 0):
        raise Exception("List of clusters to evaluate is empty")
    if(len(evaluate_labels) == 1):
        raise Exception("Single cluster provided")
    score = sklearn.metrics.silhouette_score(evaluate_node_vals, evaluate_labels, metric='cosine') 
    return score

def evaluate_clusters(g, nodes, num_clusters, random_state=100):
    labels = cluster_nodes_coclustering(g, nodes, num_clusters, random_state=100) 
    return evaluate_clustering(g, nodes, labels)

def evaluate_cluster_range(g, nodes, start, end, random_state=100):
    scores = [evaluate_clusters(g, nodes, i, random_state=random_state) for i in range(start, end)]
    return [i for i in range(start, end)], scores


