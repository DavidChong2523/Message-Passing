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

    return stats

