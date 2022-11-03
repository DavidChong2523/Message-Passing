import networkx as nx
import numpy as np

import msg_passing
import utils

HIST_GR = [
    "nra",
    "second amendment",
    "democrats",
    "national rifle association",
    "uvalde",
    "texas", 
    "republicans",
    "biden",
    "donald trump",
    "california",
    "houston",
    "ucla",
    "supreme court",
    "gavin newsom",
    #"assault weapons",
    "marco rubio",
    #"childersberg",
    #"michael jennings",
    "parkland",
    "florida",
    "new york",
    "congress",
    #"background checks",
    #"cnn",
] 

HIST_RW = [
    "clarence thomas",
    "supreme court",
    "republicans",
    "republican",
    "donald trump",
    "democrats",
    "joe biden",
    "pro-life",
    "roe v. wade",
    "white house",
    "planned parenthood",
    "kamala harris",
    "brett kavanaugh",
    #"abortion",
    "pro-abortion",
    "alexandria ocasio-cortez",
]

HIST_U = [
    "ukraine",
    "russian",
    "russia",
    "vladimir putin",
    "volodymyr zelensky",
    "kyiv",
    "moscow",
    "kremlin",
    "russian forces",
    "boris johnson",
    "nato",
    "international atomic energy agency",
    "china",
    "pentagon",
    "joe biden",
    "us",
    "european union",
    "russian invasion",
    "crimea",
]

HIST_IM = [
    #"greg abott",
    "texas", 
    "new york city", 
    "eric adams", 
    "ron desantis", 
    "muriel bowser", 
    "manuel castro", 
    "chicago", 
    "white house",
    "joe biden", 
    "republican", 
    "democratic", 
    "lori lightfoot", 
    "kamala harris", 
    "steve bannon", 
    "illegal immigrants", 
    "fox news", 
    "u.s. customs and border protection", 
    "department of social services", 
    "department of homeland security", 
    "national guard",
]

def load_and_process(fname):
    g = msg_passing.load_graph_csv(fname, clean_data=True)
    g = utils.largest_connected_component(g)
    msg_passing.initialize_node_values(g, size=3)
    return g 

def save_results(fname_prefix, g, hist, diagnostic_hist):
    msg_passing.save_graph(g, fname_prefix + ".graphml")
    msg_passing.save_history(hist, diagnostic_hist, fname_prefix + ".pkl")

def set_history(nodes):
    history = {}
    for n in nodes:
        history[n] = []
    return history

def get_print_and_save_period(total_iters, num_print, num_save):
    return total_iters // num_print, total_iters // num_save

def train_on_files(indir, infiles, outdir, out_suffix, iters, num_print, num_save): 
    for f in infiles: 
        fname = indir + f 
        outfile = outdir + f[:-len(".csv")] + out_suffix

        print("Running file: " + fname)
        print("Saving output to: " + outfile)

        g = load_and_process(fname)
        pper, sper = get_print_and_save_period(iters, num_print, num_save)
        pg, _ = msg_passing.prune_graph(g)
        top_nodes = [nd[0] for nd in utils.node_degrees(pg)[:10]]
        history = set_history(top_nodes)
        hist, diagnostic_hist = msg_passing.pass_messages_with_random_walks(
            g, 10**-3, 10**-3, iters, True, print_period=pper, save_period=sper, history=history, 
            discount=0.95, path_length=10, batch_size=10
        )
        save_results(outfile, g, hist, diagnostic_hist)

def run_incremental():
    indir = "input/Incremental_Datasets/"
    infiles = [
        "global_warming_network.csv",
        "gun_regulations_network.csv",
        "immigration_network.csv",
        "inflation_network.csv",
        "roe_v_wade_network.csv",
        "trump_impeachment_network.csv",
        "ukraine_war_network.csv",
        "vaccine_hesitancy_network.csv",
        "combined.csv"
    ]
    outdir = "output/test/"
    out_suffix = "_random_walks_lr_10-3_40K_dc_095_pl_10_bs_10"
    iters = 40000
    num_print = 100
    num_save = 1000
    train_on_files(indir, infiles, outdir, out_suffix, iters, num_print, num_save)

if __name__ == "__main__":
    run_incremental()

        

