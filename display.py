import numpy as np
import matplotlib.pyplot as plt
import copy

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
    plt.legend()
    plt.show()
    plt.plot([i for i in range(len(diagnostic_hist["LOSS"]))], diagnostic_hist["LOSS"], label="loss")
    plt.legend()
    plt.show()

# if target provided, plot cos distance between all nodes and target
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
    plt.rcParams["figure.figsize"] = (10,5)
    plt.show()
