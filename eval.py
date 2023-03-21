import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from collections import Counter, defaultdict
from config import *
from dataset import *
# from models.mnist_cnn import MNISTCNN
from models.hparam import HParam
from models.gans import *
from models.dc_gan_model import *
from utils import *
from models.ood_gan_backbone import *
from ood_gan import *
import umap
import umap.plot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def tpr(winv, woutv, level=0.95):
    assert level < 1 and level > 0
    threshold = np.quantile(winv.to('cpu'), level)
    ic(f"{level*100}% TNR Threshold: {threshold}")
    fpr = woutv[woutv <= threshold].shape[0] / float(woutv.shape[0])
    tpr = 1 - fpr
    ic(f"TPR at {level*100}%  TNR: {tpr}")
    return threshold, tpr


def logistic_regression(D, G, winv, woutv):
    pass


class EVALER():
    def __init__(self, xin_v, xout_v):
        self.xin_v = xin_v
        self.xout_v = xout_v
        self.tpr95, self.tpr99 = [], []
        self.tpr95_thresh, self.tpr99_thresh = [], []
        pass

    def compute_stats(self, D, G=None):
        winv = ood_wass_loss(torch.softmax(D(self.xin_v.to(DEVICE)), dim=-1))
        woutv = ood_wass_loss(torch.softmax(D(self.xout_v.to(DEVICE)), dim=-1))
        # Test model performance

        pass

    def class_stats(self, cls_idx):
        pass
