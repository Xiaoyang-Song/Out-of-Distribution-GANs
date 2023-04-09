from matplotlib import pyplot as plt
import numpy as np
import json
import torch
import eval
from eval import *
import pickle


# REGIME I: Balanced OoD samples
dset = "FashionMNIST"
with open("res/regime-I-results.json", 'r') as f:
    res_dict = json.load(f)
    # plot Experiment Results
    mnist = res_dict[dset]
    n, result, config = mnist.values()
    colors, markers = config.values()
    for idx, (item, val) in enumerate(result.items()):
        plt.plot(n, val, label=item, color=colors[idx], marker=markers[idx])
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.xlabel("Number of Observed OoD Samples for Each Class")
    plt.ylabel("OoD Detection Accuracy (TPR)")
    plt.title(f"Experimental Results - Regime I - {dset}")
    plt.savefig(f"res/plot/{dset}-Regime-I.png", dpi=400)
    # plt.show()
    plt.close()
