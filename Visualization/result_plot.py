from matplotlib import pyplot as plt
import numpy as np
import json
import torch
import eval
from eval import *
import pickle


# REGIME I: Balanced OoD samples

# with open("res/regime-I-results.json", 'r') as f:
#     res_dict = json.load(f)
#     # plot MNIST-Experiment Results
#     mnist = res_dict['MNIST']
#     n, result, config = mnist.values()
#     colors, markers = config.values()
#     for idx, (item, val) in enumerate(result.items()):
#         plt.plot(n, val, label=item, color=colors[idx], marker=markers[idx])
#     plt.grid(color='black', linestyle='-', linewidth=0.5)
#     plt.legend()
#     plt.xlabel("Number of Observed OoD Samples for Each Class")
#     plt.ylabel("OoD Detection Accuracy (TPR)")
#     plt.title("Experimental Results - Regime I - MNIST")
#     plt.savefig("res/plot/MNIST-Regime-I.png", dpi=400)
#     # plt.show()
#     plt.close()
evaler = EVALER(None, None, None, None, None, None, None, None, None, None)
evaler = torch.load("res/eval.pt", map_location=torch.device('cpu'))

# with open("res/eval.pkl", "r") as f:
#     evaler = EVALER(None, None, None, None, None, None, None, None, None, None)
#     evaler = pickle.load(f)
#     winv, woutv = evaler.winv, evaler.woutv
#     print(len(winv))
