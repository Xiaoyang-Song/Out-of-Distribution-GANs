
# Basic Utilities
import time
import os
import sys
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
import geomloss
from geomloss import SamplesLoss
# from geomloss.geomloss import SamplesLoss # problematic import
# Torchvision libraries
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
# Auxiliary libraries
from icecream import ic

# Device Auto-Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    ic("Hello config.py")
    ic(f"Device: {DEVICE}")
