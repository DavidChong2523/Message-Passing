import numpy as np
import matplotlib.pyplot as plt
import copy
import plotly.express as px 
import pandas as pd 
import sklearn.cluster 
from collections import defaultdict

import utils

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

def plot_diagnostic(diagnostic_hist):
    plt.plot([i for i in range(len(diagnostic_hist["UPDATE_MAG"]))], diagnostic_hist["UPDATE_MAG"], label="update mag")
    plt.title("update magnitude")
    plt.xlabel("iterations")
    plt.ylabel("update magnitude")
    plt.legend()
    plt.show()
    plt.plot([i for i in range(len(diagnostic_hist["LOSS"]))], np.array(diagnostic_hist["LOSS"]), label="loss")
    plt.title("loss")
    plt.ylabel("loss")
    plt.xlabel("iterations")
    plt.legend()
    plt.show()

# if target provided, plot angle between all nodes and target
def plot_history(history, target=None):
    disp_hist = copy.deepcopy(history)
    colors = iter_colors(len(disp_hist.keys()))

    for k in disp_hist.keys():
        for i in range(len(disp_hist[k])):
            if(target):
                disp_hist[k][i] = utils.between_angle(history[k][i], history[target][i])
            else:
                disp_hist[k][i] = utils.vec_to_angle(history[k][i])

    for i, k in enumerate(disp_hist.keys()):
        plt.plot([i for i in range(len(disp_hist[k]))], disp_hist[k], label=k, color=colors[i])
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.xlabel("iterations")
    plt.ylabel("angle with target node")
    plt.title("node values")
    plt.rcParams["figure.figsize"] = (10,5)
    plt.show()

# nodes is a list of strings describing the set of nodes to plot
def plot_final_values(g, nodes, target=None):
    colors = iter_colors(len(nodes))
    for i, n in enumerate(nodes):
        if(not target):
            val = utils.vec_to_angle(g.nodes()[n]["value"])
        else:
            val = utils.between_angle(g.nodes()[n]["value"], g.nodes()[target]["value"])
            #print(n, g.nodes()[n]["value"], g.nodes()[target]["value"], val)

        plt.scatter(i, val, label=n, color=colors[i])
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    if(not target):
        plt.title("final node values")
    else:
        plt.title(f"final angles with reference node '{target}'")
    plt.ylabel("angle with target node")
    plt.xlabel("nodes")
    plt.rcParams["figure.figsize"] = (10,5)
    plt.show()


def cos_dist_histogram(g, title=""):
    show_title = "histogram of pairwise cos distances"
    if(title != ""):
        show_title += ": " + title
    distances = [utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"]) for u, v in g.edges()]
    dist_df = pd.DataFrame({"pairwise cosine distances": distances})
    fig = px.histogram(dist_df, x="pairwise cosine distances", title=show_title)
    fig.show()
    return fig

def confusion_matrix(g, k, title=""):
    nodes = [n[0] for n in utils.node_degrees(g)[:k]]
    return node_confusion_matrix(g, nodes, title=title)

def node_confusion_matrix(g, nodes, title=""):
    confusion_matrix = np.zeros((len(nodes), len(nodes)))
    
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            confusion_matrix[i][j] = utils.cos_dist(g.nodes()[u]["value"], g.nodes()[v]["value"])
    
    nclusters = 2
    model = sklearn.cluster.SpectralCoclustering(n_clusters=nclusters, random_state=0) 
    model.fit(confusion_matrix) 
    row_inds = np.argsort(model.row_labels_)
    col_inds = np.argsort(model.column_labels_)
    out_matrix = confusion_matrix[row_inds]
    out_matrix = out_matrix[:, row_inds]
    
    row = [nodes[i] for i in row_inds]
    col = [nodes[i] for i in col_inds]
    labels = {
        "color": "cos dist",
        
    }
    show_title = "confusion matrix"
    if(title != ""):
        show_title += ": " + title
    fig = px.imshow(out_matrix, x=row, y=row, title=show_title, labels=labels)
    fig.update_layout(yaxis_nticks=len(nodes), xaxis_nticks=len(nodes))
    fig.show()
    return fig 

def plot_3D_node_values(g, nodes):
    vals = [g.nodes()[n]["value"] for n in nodes]
    x = [v[0] for v in vals]
    y = [v[1] for v in vals]
    z = [v[2] for v in vals] 
    df = {"x": x, "y": y, "z": z, "node": nodes}
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="node", title="Node values")
    fig.show()
    return fig