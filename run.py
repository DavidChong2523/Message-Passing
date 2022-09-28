import networkx as nx
import numpy as np

import msg_passing
import utils

def run():
    infile = "input/roe_v_wade_network_filtered.csv"
    total_g = msg_passing.load_graph_csv(infile)

    total_g = utils.largest_connected_component(total_g)
    g = total_g.copy()
    while(True):
        num_removed = utils.remove_auxillary_nodes(g)
        if(num_removed == 0):
            break
    
    msg_passing.initialize_node_values(g, mean=1, std=0.01, size=3)
    
    history = {}
    people = list(nx.shortest_path(g, "joe biden", "mitch mcconnell")) + ["supreme court", "thomas", "samuel alitos", "roe v wade", "hillary clinton", "kamala harris"]
    for p in people:
        history[p] = []

    history, diagnostic_hist = msg_passing.iterate(g, 10**-3, 10**-3, 1000, print_period=100, stop_thresh=10**-8, use_heat=True, history=history, save_period=100)
    
    return history, diagnostic_hist, g

def test():
    g = nx.MultiGraph()
    g.add_edge("a", "b", weight=-1)
    g.nodes()["a"]["value"] = utils.unit_vec(np.array([-0.1, 1]))
    g.nodes()["b"]["value"] = utils.unit_vec(np.array([0.1, 1]))
    history = {"a": [], "b": []}
    history, diagnostic_hist = msg_passing.iterate(g, 10**-1, 10**-1, 100, print_period=10, stop_thresh=10**-8, use_heat=True, history=history)
    return history, diagnostic_hist, g

def test2():
    infile = "input/roe_v_wade_network_filtered.csv"
    g = msg_passing.load_graph_csv(infile)
    g = utils.largest_connected_component(g)
    msg_passing.initialize_node_values(g, mean=1, std=0.01, size=3)
    
    history = {}
    people = list(nx.shortest_path(g, "joe biden", "mitch mcconnell")) + ["supreme court", "thomas", "samuel alitos", "roe v wade", "hillary clinton", "kamala harris"]
    for p in people:
        history[p] = []

    history, diagnostic_hist = msg_passing.pass_messages(g, 10**-3, 10**-3, 1000, print_period=100, stop_thresh=10**-8, use_heat=True, history=history, save_period=100)
    
    return history, diagnostic_hist, g

def test3():
    infile = "input/Streamed_Datasets/gun_regulations_network.csv"
    g = msg_passing.load_graph_csv(infile, with_dates=True)
    g = utils.largest_connected_component(g)
    msg_passing.initialize_node_values(g, mean=1, std=0.01, size=3)

    dates = msg_passing.generate_graph_date_range(g, 90)
    history = {}
    iters = 100
    save_period = iters // 100
    save_path = "output/gun_regulations_test/test"
    results = msg_passing.pass_messages_time_series(dates, g, 10**-3, 10**-3, iters, print_period=None, stop_thresh=10**-8, use_heat=True, history=history, save_period=save_period, remove_undated_edges=True, save_path=save_path)
    return dates, results

if __name__ == "__main__":
    hist, diagnostic_hist, g = run()
    outfile = "test.graphml"
    msg_passing.save_graph(g, outfile)


        

