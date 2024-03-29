from matplotlib import pyplot as plt
import numpy as np
import json
import torch
# import eval
# from eval import *
import pickle


# REGIME I: Balanced OoD samples
# dset = "FashionMNIST"
dset = "MNIST"
regime = "I"
with open(f"res/regime-{regime}-results.json", 'r') as f:
    res_dict = json.load(f)
    # plot Experiment Results
    mnist = res_dict[dset]
    n, result, std, config = mnist.values()
    labels, colors, markers, linestyles = config.values()
    # plt.figure(figsize=(8, 6))
    for idx, (item, val) in enumerate(result.items()):
        if type(val) == list:
            vals = val
        else:
            vals = [val] * n
        plt.plot(n, vals, label=labels[idx], linestyle=linestyles[idx],
                 color=colors[idx], marker=markers[idx], markersize=4)
        # plt.fill_between(
        #     n, val-np.array(std[item])*100, val+np.array(std[item])*100, alpha=0.10, color=colors[idx])
        # plt.errorbar(n, val, yerr=np.array(std[item]) * 100, label=item,
        #              color=colors[idx], marker=markers[idx], capsize=3, markersize=4)
        # if idx == 1:
        #     break
    # plt.grid(color='black', linestyle='-', linewidth=0.5)
    # plt.axhline(y=odin['95'], label="ODIN", linestyle="solid", color="")
    # plt.axhline(y=odin['99'], linestyle="dashed", color="green")

    plt.legend(loc=4)
    # plt.xlabel("Number of Observed OoD Samples for Each Class")
    plt.xlabel("Number of Observed OoD Samples for Selected Classes")
    plt.ylabel("OoD Detection Accuracy (TPR)")
    plt.title(f"Experimental Results - Regime {regime} - {dset}")
    plt.savefig(f"res/plot/{dset}-Regime-{regime}.png", dpi=1000)
    # plt.show()
    plt.close()
