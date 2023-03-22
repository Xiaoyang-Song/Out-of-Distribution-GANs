from email.policy import default
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


class LR():
    def __init__(self, D, G, xin_t, n=10, C=5):
        self.D, self.G = D, G
        self.xin, _ = tuple_list_to_tensor(xin_t)
        self.xin = self.xin[np.random.choice(len(self.xin), n), :, :, :]
        self.n, self.c = n, C
        # Statistics
        self.train_stats = None

    def fit(self):
        yin = torch.ones(len(self.xin))
        # Generate OoD images
        g_seed = sample_noise((self.n // self.c) * self.c, 96)
        n_class = self.n // self.c
        gz = self.G(g_seed, np.array(
            [[i] * n_class for i in range(5)]).flatten())
        yz = torch.zeros(n_class * 5)
        # Form training dataset
        win = ood_wass_loss(torch.softmax(self.D(self.xin.to(DEVICE)), dim=-1))
        wgz = ood_wass_loss(torch.softmax(self.D(gz.to(DEVICE)), dim=-1))
        mean_win, mean_wgz = torch.mean(wgz), torch.mean(wgz)
        ic(f"Mean win {mean_win} ; Mean wgz {mean_wgz}")
        # Training
        x = torch.cat([win, wgz])
        y = torch.ones(len(yz) + len(yin))
        y[0:len(yin)] = 0  # InD = 0; OoD = 1
        X, Y = x.unsqueeze(-1).data.cpu(), y.data.cpu()
        clf = LogisticRegression(random_state=0).fit(X, Y)
        training_acc = clf.score(X, Y)
        ic(f"Training accuracy: {training_acc}")
        self.train_stats = [mean_win, mean_wgz, training_acc]
        self.clf = clf
        return [mean_win, mean_wgz, training_acc]

    def eval(self, winv, woutv):
        assert self.clf is not None
        ind_acc = self.clf.score(
            winv.unsqueeze(-1).data.cpu(), np.zeros(len(winv)))
        ood_acc = self.clf.score(
            woutv.unsqueeze(-1).data.cpu(), np.ones(len(woutv)))
        ic(f"Testing accuracy on InD: {ind_acc}")
        ic(f"Testing accuracy on OoD: {ood_acc}")
        # Calculate AUROC
        in_prob = self.clf.predict_proba(winv.unsqueeze(-1).data.cpu())
        out_prob = self.clf.predict_proba(woutv.unsqueeze(-1).data.cpu())
        auroc = roc_auc_score([1]*len(woutv) + [0]*len(winv),
                              list(out_prob[:, 1])+list(in_prob[:, 1]))
        ic(f"Testing AUROC: {auroc}")
        self.eval_stats = [ind_acc, ood_acc, auroc]
        return [ind_acc, ood_acc, auroc]


class EVALER():
    def __init__(self, xin_t, xin_v, xout_v):
        self.xin_t = xin_t  # InD training dataset
        self.xin_v = xin_v  # InD Testing dataset
        self.xout_v = xout_v  # OoD Testing dataset
        # Statistics - TPR at x% TNR
        self.tpr95, self.tpr99 = [], []
        self.tpr95_thresh, self.tpr99_thresh = [], []
        # Statistics - Logistic Regression
        # Refer to the return values of logistic regression for details.
        self.lr_instance = []
        self.lr_train = []
        self.lr_overall = []
        # Refer to class_level statistics
        self.cls_stats = defaultdict(list)

    def compute_stats(self, D, G=None, each_class=False, cls_idx=None):
        xinv, yxinv = tuple_list_to_tensor(self.xin_v)
        xoutv, yxoutv = tuple_list_to_tensor(self.xout_v)
        winv = ood_wass_loss(torch.softmax(D(xinv.to(DEVICE)), dim=-1))
        woutv = ood_wass_loss(torch.softmax(D(xoutv.to(DEVICE)), dim=-1))
        # Test model performance
        tpr_95, tpr_95_thresh = tpr(winv, woutv, 0.95)
        tpr_99, tpr_99_thresh = tpr(winv, woutv, 0.99)
        self.tpr95.append(tpr_95)
        self.tpr95_thresh.append(tpr_95_thresh)
        self.tpr99.append(tpr_99)
        self.tpr99_thresh.append(tpr_99_thresh)
        if G is not None:
            lr = LR(D, G, self.xin_t)
            train_stats = lr.fit()
            eval_stats = lr.eval(winv, woutv)
            self.lr_instance.append(lr)
            self.lr_train.append(train_stats)
            self.lr_overall.append(eval_stats)
        if each_class:
            assert cls_idx is not None
            for idx in cls_idx:
                ic(f"Class: {idx}")
                mask = yxoutv == cls_idx
                woutv_idx = woutv[mask]
                tpr_95, tpr_95_thresh = tpr(winv, woutv_idx, 0.95)
                tpr_99, tpr_99_thresh = tpr(winv, woutv_idx, 0.99)
                result = [tpr_95, tpr_95_thresh, tpr_99, tpr_99_thresh]
                if G is not None:
                    cls_lr_eval = lr.eval(winv, woutv_idx)
                    result += cls_lr_eval
                self.cls_stats[idx].append(result)

    def display_stats(self):
        # Overall stats
        ic("Overall Statistics")
        ic(self.tpr95)
        ic(f"mean: {np.round(np.mean(self.tpr95), 5)} | std: {np.round(np.std(self.tpr95), 5)}")
        ic(self.tpr99)
        ic(f"mean: {np.round(np.mean(self.tpr99), 5)} | std: {np.round(np.std(self.tpr99), 5)}")
        ic(self.tpr95_thresh)
        ic(f"mean: {np.round(np.mean(self.tpr95_thresh), 5)} | std: {np.round(np.std(self.tpr95_thresh), 5)}")
        ic(self.tpr99_thresh)
        ic(f"mean: {np.round(np.mean(self.tpr99_thresh), 5)} | std: {np.round(np.std(self.tpr99_thresh), 5)}")
        if len(self.lr_instance) != 0:
            lr_train = np.array(self.lr_train)
            ic(f"mean_win: {lr_train[:,0]}")
            ic(f"mean: {np.round(np.mean(lr_train[:,0]), 5)} | std: {np.round(np.std(lr_train[:,0]), 5)}")
            ic(f"mean_wgz: {lr_train[:,1]}")
            ic(f"mean: {np.round(np.mean(lr_train[:,1]), 5)} | std: {np.round(np.std(lr_train[:,1]), 5)}")
            ic(f"training_acc: {lr_train[:,2]}")
            ic(f"mean: {np.round(np.mean(lr_train[:,2]), 5)} | std: {np.round(np.std(lr_train[:,2]), 5)}")
            lr_stats = np.array(self.lr_overall)
            ic(f"ind_acc: {lr_stats[:,0]}")
            ic(f"mean: {np.round(np.mean(lr_stats[:,0]), 5)} | std: {np.round(np.std(lr_stats[:,0]), 5)}")
            ic(f"ood_acc: {lr_stats[:,1]}")
            ic(f"mean: {np.round(np.mean(lr_stats[:,1]), 5)} | std: {np.round(np.std(lr_stats[:,1]), 5)}")
            ic(f"auroc: {lr_stats[:,2]}")
            ic(f"mean: {np.round(np.mean(lr_stats[:,2]), 5)} | std: {np.round(np.std(lr_stats[:,2]), 5)}")
        if len(self.cls_stats) != 0:
            for key, val in self.cls_stats.items():
                ic(f"Class: {key}")
                vals = np.array(val)
                ic(f"tpr95: {vals[:,0]}")
                ic(f"mean: {np.round(np.mean(vals[:,0]), 5)} | std: {np.round(np.std(vals[:,0]), 5)}")
                ic(f"tpr99: {vals[:,2]}")
                ic(f"mean: {np.round(np.mean(vals[:,2]), 5)} | std: {np.round(np.std(vals[:,2]), 5)}")
                if len(val) > 4:
                    lr_stats = vals[4:7]
                    ic(f"ind_acc: {lr_stats[:,0]}")
                    ic(f"mean: {np.round(np.mean(lr_stats[:,0]), 5)} | std: {np.round(np.std(lr_stats[:,0]), 5)}")
                    ic(f"ood_acc: {lr_stats[:,1]}")
                    ic(f"mean: {np.round(np.mean(lr_stats[:,1]), 5)} | std: {np.round(np.std(lr_stats[:,1]), 5)}")
                    ic(f"auroc: {lr_stats[:,2]}")
                    ic(f"mean: {np.round(np.mean(lr_stats[:,2]), 5)} | std: {np.round(np.std(lr_stats[:,2]), 5)}")
