import networkx as nx
import numpy as np
import re

import constants

def largest_connected_component(g):
    largest_cc = max(nx.connected_components(g), key=len)
    subgraph = g.subgraph(largest_cc)
    return subgraph

def ego_network(g, n, radius=1):
    eg = nx.ego_graph(g, n, radius=radius)
    return eg

# remove nodes with one neighbor
def remove_auxiliary_nodes(g):
    to_remove = [n for n in g.nodes() if len(list(g.neighbors(n))) <= 1]
    g.remove_nodes_from(to_remove)
    return to_remove, len(to_remove)

def get_ques_val(ques):
    for q, v in constants.QUES_VAL.items():
        if(re.match(q, ques, re.IGNORECASE)):
            return v
    return None

# angle [0, 2pi) to unit vector
def angle_to_vec(angle):
    return np.array([np.cos(angle), np.sin(angle)])

# 2D vector to angle from x-axis
def vec_to_angle(vec):
    return np.arctan2(vec[1], vec[0]) % (2*np.pi)

def unit_vec(vec):
    if(not vec.any()):
        return vec
    return vec / np.linalg.norm(vec)

# unit vector that is average of given vectors
# input is numpy array of vectors with shape (num_vecs, vec_dim), num_vecs > 0
# weights is a numpy array of weights with shape (vec_dim,)
def avg_vec(vecs, weights=np.array([])):
    if(len(weights) == 0):
        weights = (1/vecs.shape[-1])*np.ones(vecs.shape[-1])
    weights = weights[:, np.newaxis]
    avg = np.sum(vecs*weights, axis=0)
    return unit_vec(avg)

# angle between v1 and v2
# v1 and v2 have shape (dim,)
def between_angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos, -1, 1))

# grad of 1 - (dot(x, y) / (norm(x)*norm(y)))
# grad is -1/(norm(x)*norm(y)) * [norm(x)y - dot(x, y)x/norm(x)]
# inputs must be unit vectors
def vec_grad(x, y):
    return np.dot(x, y)*x - y

# inputs must be unit vectors
def cos_dist(x, y):
    cos_dist = 1 - np.dot(x, y) 
    return cos_dist

def average_degree(g):
    degrees = [g.degree[n] for n in g.nodes()]
    avg_degree = sum(degrees) / len(degrees)
    return avg_degree

def average_value(g):
    vals = [g.nodes()[n]["value"] for n in g.nodes()]
    return sum(vals) / len(vals)
