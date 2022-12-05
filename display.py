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

"""
Default Plotly Colors: https://community.plotly.com/t/plotly-colours-list/11730/3 
"""
COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

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

def plot_diagnostic(diagnostic_hist, title=None, show=True):
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

    plt_title = "Diagnostic History" 
    plt_title = generate_title(plt_title, title)
    fig.update_layout(showlegend=False, title_text=plt_title)
    if(show):
        fig.show()
    return fig

# TODO: left align subplot titles - https://community.plotly.com/t/subplot-title-alignment/33210/2
def plot_diagnostic_grid(hist_files, titles, plt_title_text, plt_rows, plt_cols, show=True):
    plt_titles = [titles[i // 2] if i % 2 == 0 else "" for i in range(len(titles)*2)]
    fig = make_subplots(rows=plt_rows, cols=plt_cols*2, subplot_titles=plt_titles, x_title="Iterations")
    for i, f in enumerate(hist_files):
        dh = msg_passing.load_history(f)[1]
        dh_row = (i // plt_cols) + 1
        dh_col = (i % plt_cols) + 1
        update_mag_col = dh_col*2-1
        loss_col = dh_col*2
        update_mag_showlegend = True if i == 0 else False
        loss_showlegend = True if i == 0 else False

        iters = np.array([i for i in range(len(dh["LOSS"]))])
        if(dh["SAVE_PER"]):
            iters *= dh["SAVE_PER"][0]
        fig.append_trace(
            go.Scatter(
                x=iters,
                y=dh["UPDATE_MAG"],
                line=dict(color=COLORS[0]),
                name="Update Magnitude",
                showlegend=update_mag_showlegend
            ), row=dh_row, col=update_mag_col
        )
        fig.append_trace(
            go.Scatter(
                x=iters,
                y=dh["LOSS"],
                line=dict(color=COLORS[3]),
                name="Loss",
                showlegend=loss_showlegend
            ), row=dh_row, col=loss_col
        )
        #fig.update_yaxes(title_text="Update Magnitude", row=dh_row, col=update_mag_col)
        #fig.update_yaxes(title_text="Loss", row=dh_row, col=loss_col)
        

    fig.update_layout(showlegend=True, title_text=plt_title_text)
    if(show):
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


def plot_cos_dist_histogram_grid(graph_files, titles, plt_title_text, plt_rows, plt_cols, random_baseline=False, show=True):
    fig = make_subplots(rows=plt_rows, cols=plt_cols, subplot_titles=titles, y_title="Count (log scale)", x_title="Pairwise Cosine Distances")
    for i, f in enumerate(graph_files):
        g = msg_passing.load_graph_graphml(f)
        g, _ = msg_passing.prune_graph(g)
        if(random_baseline):
            msg_passing.initialize_node_values(g, g.nodes()[list(g)[0]]["value"].shape[0])

        g_row = (i // plt_cols) + 1
        g_col = (i % plt_cols) + 1
        distances = [utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"]) for u, v in g.edges()]

        # number of bins: https://medium.datadriveninvestor.com/how-to-decide-on-the-number-of-bins-of-a-histogram-3c36dc5b1cd8 
        num_bins = int(1 + np.ceil(np.log2(len(distances))))
        max_dist = 2
        step = max_dist / num_bins
        counts = np.zeros(num_bins)
        bins = [(step/2) + step*n for n in range(num_bins)]
        for d in distances:
            if(d >= 2):
                counts[-1] += 1
            else:
                counts[int(max(d, 0) // step)] += 1

        fig.append_trace(
            go.Bar(
                x=bins,
                y=counts,
                width=step,
                marker_color="blue"
            ), row=g_row, col=g_col
        )
        fig.update_yaxes(type="log")
        fig.update_xaxes(range=[0, 2])
    
    fig.update_layout(showlegend=False, title_text=plt_title_text)
    if(show):
        fig.show()
    return fig

def plot_confusion_matrix(g, nodes, num_clusters, title=None, show=True):
    def truncate_row_names(row, max_len):
        return [r if len(r) < max_len else r[:max_len-3] + "..." for r in row]

    row, confusion_matrix = parse_data.generate_clustered_confusion_matrix(g, nodes, num_clusters)
    row = truncate_row_names(row, 20)
    labels = {
        "color": "Cosine Distance",
    }

    plt_title = "Confusion Matrix"
    plt_title = generate_title(plt_title, title)
    fig = px.imshow(confusion_matrix, x=row, y=row, labels=labels, title=plt_title)
    fig.update_layout(yaxis_nticks=len(nodes), xaxis_nticks=len(nodes))
    if(show):
        fig.show()
    return fig 

def plot_confusion_matrix_with_random_baseline(g, nodes, num_clusters, title=None, show=True):
    def truncate_row_names(row, max_len):
        return [r if len(r) < max_len else r[:max_len-3] + "..." for r in row]

    plt_title = generate_title("Confusion Matrix", title)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Learned Representations", "Random Baseline"))
    row, confusion_matrix = parse_data.generate_clustered_confusion_matrix(g, nodes, num_clusters)
    row = truncate_row_names(row, 30)

    rand_g = g.copy()
    msg_passing.initialize_node_values(rand_g, size=g.nodes()[list(g)[0]]["value"].shape[0])
    row_baseline, confusion_matrix_baseline = parse_data.generate_clustered_confusion_matrix(rand_g, nodes, num_clusters)
    row_baseline = truncate_row_names(row_baseline, 15)

    fig.add_trace(go.Heatmap(x=row, y=row[::-1], z=confusion_matrix[::-1, :], colorbar=dict(title='Cosine Distance')), row=1, col=1)
    fig.add_trace(go.Heatmap(x=row_baseline, y=row_baseline[::-1], z=confusion_matrix_baseline[::-1, :], showscale=False), row=1, col=2)
    fig.update_layout(yaxis_nticks=len(nodes), xaxis_nticks=len(nodes), title_text=plt_title)
    if(show):
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

# Note: top_n_nodes is [str] where each element is a str(number) or 'all', which takes all nodes
def plot_top_n_cluster_evals(g, top_n_nodes, start, end, plt_title, show=True):
    fig = go.Figure()
    plt_x = [i for i in range(start, end)]
    for i, str_n in enumerate(top_n_nodes):
        plt_n = "top " + str_n + " nodes"
        scores = []
        random_scores = []
        errors = []
        for num_clusters in plt_x:
            rand_g = g.copy()
            if(str_n != "all"):
                n = int(str_n)
            else:
                n = len(g.nodes())

            nodes = utils.get_top_n_nodes(g, n)
            scores.append(parse_data.evaluate_clusters(g, nodes, num_clusters))

            random_sample_size = 5
            random_sample_scores = []
            for _ in range(random_sample_size):
                msg_passing.initialize_node_values(rand_g, size=g.nodes()[list(g)[0]]["value"].shape[0])
                random_sample_scores.append(parse_data.evaluate_clusters(rand_g, nodes, num_clusters))
            random_mean = np.mean(np.array(random_sample_scores))
            random_error = (max(random_sample_scores) - min(random_sample_scores)) / 2
            random_scores.append(random_mean)
            errors.append(random_error)
        
        fig.add_trace(go.Scatter(
            x=plt_x, 
            y=scores,
            mode="lines+markers",
            marker_symbol="circle",
            marker_color=COLORS[i],
            line=go.scatter.Line(color=COLORS[i], dash="dash"),
            name=plt_n,
        ))

        fig.add_trace(go.Scatter(
            x=plt_x,
            y=random_scores,
            mode="lines+markers",
            marker_symbol="diamond",
            marker_color=COLORS[i],
            line=go.scatter.Line(color=COLORS[i], dash="dot"),
            error_y=go.scatter.ErrorY(array=errors, color=COLORS[i]),
            name=plt_n + ", random baseline"
        ))
                
    fig.update_layout(
        xaxis_title="Number of Clusters", yaxis_title="Silhouette Score", title_text=plt_title)
    if(show):
        fig.show()
    return fig

# Note: top_n_nodes is [str] where each element is a str(number) or 'all', which takes all nodes
def plot_k_cluster_evals(graph_files, issues, top_n_nodes, num_clusters, plt_title, show=True):
    fig = go.Figure()
    for i, str_n in enumerate(top_n_nodes):
        plt_n = "top " + str_n + " nodes"
        scores = []
        random_scores = []
        errors = []
        for f in graph_files:
            g = msg_passing.load_graph_graphml(f) 
            pg, _ = msg_passing.prune_graph(g)
            rand_g = pg.copy()
            if(str_n != "all"):
                n = int(str_n)
            else:
                n = len(pg.nodes())

            nodes = utils.get_top_n_nodes(pg, n)
            scores.append(parse_data.evaluate_clusters(pg, nodes, num_clusters))

            random_sample_size = 5
            random_sample_scores = []
            for _ in range(random_sample_size):
                msg_passing.initialize_node_values(rand_g, size=pg.nodes()[list(pg)[0]]["value"].shape[0])
                random_sample_scores.append(parse_data.evaluate_clusters(rand_g, nodes, num_clusters))
            random_mean = np.mean(np.array(random_sample_scores))
            random_error = (max(random_sample_scores) - min(random_sample_scores)) / 2
            random_scores.append(random_mean)
            errors.append(random_error)
        
        fig.add_trace(go.Scatter(
            x=issues, 
            y=scores,
            mode="lines+markers",
            marker_symbol="circle",
            marker_color=COLORS[i],
            line=go.scatter.Line(color=COLORS[i], dash="dash"),
            name=plt_n,
        ))

        fig.add_trace(go.Scatter(
            x=issues,
            y=random_scores,
            mode="lines+markers",
            marker_symbol="diamond",
            marker_color=COLORS[i],
            line=go.scatter.Line(color=COLORS[i], dash="dot"),
            error_y=go.scatter.ErrorY(array=errors, color=COLORS[i]),
            name=plt_n + ", random baseline"
        ))
                
    fig.update_layout(
        xaxis_title="Issues", yaxis_title="Silhouette Score", title_text=plt_title)
    if(show):
        fig.show()
    return fig



def plot_edge_weight_histogram(g, title=None, log_scale=False, show=True):
    edge_weights = [w for _, _, w in g.edges(data="weight")]
    df = {"Edge Weight": edge_weights}

    plt_title = "Histogram of Edge Weights"
    plt_title = generate_title(plt_title, title)
    fig = px.histogram(df, x="Edge Weight", title=plt_title)
    if(log_scale):
        fig.update_yaxes(type="log")
    if(show):
        fig.show()
    return fig

def plot_degree_histogram(g, title=None, log_scale=False, show=True):
    degrees = [g.degree[n] for n in g.nodes()]
    df = {"Node Degree": degrees}

    plt_title = "Histogram of Node Degrees"
    plt_title = generate_title(plt_title, title)
    fig = px.histogram(df, x="Node Degree", title=plt_title)
    if(log_scale):
        fig.update_yaxes(type="log")
    if(show):
        fig.show()
    return fig
