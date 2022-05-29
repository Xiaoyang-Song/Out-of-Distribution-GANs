# Basic Utilities
import time
import os
import sys
import pickle
# Basic computation libraries
# import pykeops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
# Torch-related libraries
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
from torch import linalg as LA
import geomloss
from geomloss import SamplesLoss
# from geomloss.geomloss import SamplesLoss # problematic import
# Torchvision libraries
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.gridspec as gridspec
# Auxiliary libraries
from icecream import ic
import enum
from enum import Enum, unique

# Device Auto-Configuration (Compatible with Colab)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# PATH
MODEL_SAVE_PATH = "./mnist_cnn.pt"
GAN_SAVE_PATH = "./gans_imgs"
# GANs constant
NOISE_DIM = 96
if __name__ == "__main__":
    ic("Hello config.py")
    ic(f"Device: {DEVICE}")
