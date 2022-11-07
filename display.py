import numpy as np
import matplotlib.pyplot as plt
import copy
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd 
import sklearn.cluster 
from collections import defaultdict

import parse_data
import utils
import msg_passing

def iter_colors(num_labels):
    def hsv_to_rgb(hsv):
        h, s, v = hsv
        c = s*v
        h_prime = h / 60
        x = c*(1 - abs(h_prime % 2 - 1))
        if(h_prime >= 0 and h_prime < 1):
            r1, g1, b1 = c, x, 0
        elif(h_prime >= 1 and h_prime < 2):
            r1, g1, b1 = x, c, 0
        elif(h_prime >= 2 and h_prime < 3):
            r1, g1, b1 = 0, c, x
        elif(h_prime >= 3 and h_prime < 4):
            r1, g1, b1 = 0, x, c
        elif(h_prime >= 4 and h_prime < 5):
            r1, g1, b1 = x, 0, c
        elif(h_prime >= 5 and h_prime < 6):
            r1, g1, b1 = c, 0, x
        m = v - c
        r, g, b = r1 + m, g1 + m, b1 + m
        return (r, g, b)
    vals = np.linspace(0, 360, num=num_labels, endpoint=False)
    rgb_vals = [hsv_to_rgb((v, 0.75, 1)) for v in vals]
    return rgb_vals

def generate_title(base, suffix):
    if(suffix):
        return base + ": " + suffix 
    return base

def plot_diagnostic(diagnostic_hist):
    iters = np.array([i for i in range(len(diagnostic_hist["LOSS"]))])
    if(diagnostic_hist["SAVE_PER"]):
        iters *= diagnostic_hist["SAVE_PER"][0]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Update Magnitude", "Loss"), x_title="Iterations") 
    fig.append_trace(
        go.Scatter(
            x=iters,
            y=diagnostic_hist["UPDATE_MAG"]
        ), row=1, col=1
    )
    fig.append_trace(
        go.Scatter(
            x=iters,
            y=diagnostic_hist["LOSS"]
        ), row=1, col=2
    )
    fig.update_layout(showlegend=False, title_text="Diagnostic History")
    fig.show()
    return fig

# if target provided, plot angle between all nodes and reference
def plot_history_with_reference(history, reference=None):
    cos_dists, iters, nodes = [], [], []
    for node, vals in history.items():
        for i, v in enumerate(vals):
            cos_dists.append(utils.cos_dist(v, history[reference][i]))
            iters.append(i)
            nodes.append(node) 

    df = pd.DataFrame({"Cosine Distance to Reference Node": cos_dists, "Iteration": iters, "Node": nodes})
    plt_title = "History of Node Values with Reference Node: " + reference
    fig = px.line(df, x="Iteration", y="Cosine Distance to Reference Node", color="Node", title=plt_title)
    fig.show()
    return fig

# requires 3d node values
def plot_3D_node_values(g, nodes, title=None):
    vals = [g.nodes()[n]["value"] for n in nodes]
    x = [v[0] for v in vals]
    y = [v[1] for v in vals]
    z = [v[2] for v in vals] 
    df = {"x": x, "y": y, "z": z, "node": nodes}

    plt_title = "Node Values"
    plt_title = generate_title(plt_title, title)
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="node", title=plt_title)
    fig.show()
    return fig

def plot_cos_dist_histogram(g, title=None):
    distances = [utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"]) for u, v in g.edges()]
    df = {"Pairwise Cosine Distances": distances}

    plt_title = "Histogram of Pairwise Cosine Distances"
    plt_title = generate_title(plt_title, title)
    fig = px.histogram(df, x="Pairwise Cosine Distances", title=plt_title)
    fig.show()
    return fig

def plot_confusion_matrix(g, nodes, num_clusters, title=None):
    row, confusion_matrix = parse_data.generate_clustered_confusion_matrix(g, nodes, num_clusters)
    labels = {
        "color": "Cosine Distance",
    }

    plt_title = "Confusion Matrix"
    plt_title = generate_title(plt_title, title)
    fig = px.imshow(confusion_matrix, x=row, y=row, labels=labels, title=plt_title)
    fig.update_layout(yaxis_nticks=len(nodes), xaxis_nticks=len(nodes))
    fig.show()
    return fig 

def plot_cluster_evaluations(g, nodes, start, end, title=None, show=True):
    num_clusters, scores = parse_data.evaluate_cluster_range(g, nodes, start, end)
    df = {"Number of Clusters": num_clusters, "Silhouette Score": scores}

    plt_title = "Cluster Evaluations"
    plt_title = generate_title(plt_title, title)
    fig = px.line(df, x="Number of Clusters", y="Silhouette Score", title=plt_title)
    if(show):
        fig.show()
    return fig

# random baseline only works with size 3 node vectors
def plot_top_n_cluster_evaluations(g, num_nodes_list, start, end, title=None, with_random_baseline=False, show=True):
    if(with_random_baseline):
        rand_g = g.copy()
        msg_passing.initialize_node_values(rand_g, size=3)

    all_num_clusters, all_scores, all_top_n_nodes, all_is_random = [], [], [], []
    for n in num_nodes_list:
        num_clusters, scores = parse_data.evaluate_cluster_range(g, utils.get_top_n_nodes(g, n), start, end)
        top_n_nodes = [n for _ in num_clusters] 
        is_random = [False for _ in num_clusters]
        if(with_random_baseline): 
            num_clusters_random, scores_random = parse_data.evaluate_cluster_range(
                rand_g, utils.get_top_n_nodes(g, n), start, end)
            top_n_nodes_random = [n for _ in num_clusters]
            is_random_random = [True for _ in num_clusters] 
            num_clusters += num_clusters_random 
            scores += scores_random
            top_n_nodes += top_n_nodes_random
            is_random += is_random_random
        all_num_clusters += num_clusters
        all_scores += scores
        all_top_n_nodes += top_n_nodes
        all_is_random += is_random

    df = pd.DataFrame({
        "Number of Clusters": all_num_clusters, 
        "Silhouette Score": all_scores, 
        "Top N Nodes Clustered": all_top_n_nodes,
        "Is Random Baseline": all_is_random
    })

    plt_title = "Cluster Evaluations"
    plt_title = generate_title(plt_title, title)
    fig = px.line(df, x="Number of Clusters", y="Silhouette Score", color="Top N Nodes Clustered", line_dash="Is Random Baseline", title=plt_title)
    if(show):
        fig.show()
    return fig
