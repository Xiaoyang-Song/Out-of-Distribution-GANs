from email.policy import default
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
    print(f"{level*100}% TNR Threshold: {threshold}")
    fpr = woutv[woutv <= threshold].shape[0] / float(woutv.shape[0])
    tpr = 1 - fpr
    print(f"TPR at {level*100}%  TNR: {tpr}")
    return threshold, tpr


def plot_wass_dist_and_thresh(wass_lst, legend_lst, n_ood, log_dir, tag,
                              thresh_lst=None, thresh_lbl_lst=None, bins=50, alpha=0.5):
    assert len(wass_lst) == len(legend_lst)
    for wass, lbl in zip(wass_lst, legend_lst):
        plt.hist(wass.cpu().numpy(), bins=bins, alpha=alpha, label=lbl)
    # for thresh, thresh_lbl in zip(thresh_lst, thresh_lbl_lst):
    #     plt.axvline(x=thresh, color='b', label=thresh_lbl)
    plt.legend()
    plt.xlabel("Wasserstein Distance")
    plt.ylabel("Number of Samples")
    plt.title(f"Number of observed OoD: {n_ood}")
    plt.savefig(log_dir + f"wass_plot[{n_ood}]-[{tag}].png")
    plt.close()


def loader_wass(data_loader, D):
    wass_dists = []
    # ic(DEVICE)
    # assert DEVICE == 'cuda'
    for (img, _) in tqdm(data_loader):
        out = D(img.to(DEVICE))
        wass_dist = ood_wass_loss(torch.softmax(out, dim=-1))
        wass_dists.append(wass_dist)
    return torch.cat(wass_dists, dim=0)


class LR():
    def __init__(self, D, G, xin_t, C, n):
        self.D, self.G = D, G
        self.xin_t, _ = tuple_list_to_tensor(xin_t)
        # self.xin_t = xin_t
        self.xin_t = self.xin_t[np.random.choice(len(self.xin_t), n), :, :, :]
        self.n, self.c = n, C
        # Statistics
        self.train_stats = None

    def fit(self):
        print("Logistic Regression Training Starts...")
        yin = torch.ones(len(self.xin_t))
        ind_loader = set_to_loader(list(zip(self.xin_t, yin)), 256, True)
        # Generate OoD images
        g_seed = sample_noise((self.n // self.c) * self.c, 96)
        n_class = self.n // self.c
        # ic(self.c)
        gz = self.G(g_seed, np.array(
            [[i] * n_class for i in range(self.c)]).flatten())
        yz = torch.zeros(n_class * self.c)
        gz_loader = set_to_loader(list(zip(gz, yz)), 256, True)
        # Form training dataset
        ic("> Evaluating InD Wasserstein distances...")
        win = loader_wass(ind_loader, self.D)
        ic("> Evaluating G(z) Wasserstein distances...")
        wgz = loader_wass(gz_loader, self.D)
        mean_win, mean_wgz = torch.mean(win), torch.mean(wgz)
        print(f"Mean win {mean_win} ; Mean wgz {mean_wgz}")
        # Training
        x = torch.cat([win, wgz])
        y = torch.ones(len(yz) + len(yin))
        y[0:len(yin)] = 0  # InD = 0; OoD = 1
        X, Y = x.unsqueeze(-1).data.cpu(), y.data.cpu()
        clf = LogisticRegression(random_state=0).fit(X, Y)
        training_acc = clf.score(X, Y)
        print(f"Training accuracy: {training_acc}")
        self.train_stats = [mean_win, mean_wgz, training_acc]
        self.clf = clf
        return [mean_win.cpu(), mean_wgz.cpu(), training_acc]

    def eval(self, winv, woutv):
        print("Logistic Regression Evaluation")
        assert self.clf is not None
        ind_acc = self.clf.score(
            winv.unsqueeze(-1).data.cpu(), np.zeros(len(winv)))
        ood_acc = self.clf.score(
            woutv.unsqueeze(-1).data.cpu(), np.ones(len(woutv)))
        print(f"Testing accuracy on InD: {ind_acc}")
        print(f"Testing accuracy on OoD: {ood_acc}")
        # Calculate AUROC
        in_prob = self.clf.predict_proba(winv.unsqueeze(-1).data.cpu())
        out_prob = self.clf.predict_proba(woutv.unsqueeze(-1).data.cpu())
        auroc = roc_auc_score([1]*len(woutv) + [0]*len(winv),
                              list(out_prob[:, 1])+list(in_prob[:, 1]))
        print(f"Testing AUROC: {auroc}")
        self.eval_stats = [ind_acc, ood_acc, auroc]
        return [ind_acc, ood_acc, auroc]


def print_stats(stat, name, precision=5):
    print(f"{name}: {stat}")
    print(
        f"mean: {np.round(np.mean(stat), precision)} | std: {np.round(np.std(stat), precision)}")


def ic_stats(stat, precision=5):
    ic(stat)
    ic(f"mean: {np.round(np.mean(stat), precision)} | std: {np.round(np.std(stat), precision)}")


class EVALER():
    def __init__(self, xin_t, xin_v, xin_v_loader, xout_v, xout_v_loader,
                 n_ood, log_dir, method, num_classes, n_lr=2000):
        self.n_ood = n_ood
        self.log_dir = log_dir
        # DATASETS
        self.xin_t = xin_t  # InD training dataset & loader
        # self.xin_t_loader = xin_t_loader
        self.xin_v = xin_v  # InD Testing dataset & loader
        self.xin_v_loader = xin_v_loader
        self.xout_v = xout_v  # OoD Testing dataset & loader
        self.xout_v_loader = xout_v_loader
        # METHODOLOGY
        self.method = method
        self.num_classes = num_classes
        self.n_lr = n_lr
        # Statistics - wasserstein distance
        self.winv, self.woutv = [],  []
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

    def evaluate(self, D, tag, G=None, each_class=False, cls_idx=None):
        ic("Computing evaluation statistics...")
        _, yxoutv = tuple_list_to_tensor(self.xout_v)
        ic("> Evaluating InD Wasserstein distances...")
        winv = loader_wass(self.xin_v_loader, D)
        ic("> Evaluating OoD Wasserstein distances...")
        woutv = loader_wass(self.xout_v_loader, D)
        self.winv.append(winv)
        self.woutv.append(woutv)
        # Test model performance
        tpr_95, tpr_95_thresh = tpr(winv, woutv, 0.95)
        tpr_99, tpr_99_thresh = tpr(winv, woutv, 0.99)
        self.tpr95.append(tpr_95)
        self.tpr95_thresh.append(tpr_95_thresh)
        self.tpr99.append(tpr_99)
        self.tpr99_thresh.append(tpr_99_thresh)
        yxoutv = None
        winv, woutv = None, None
        if self.method == "OOD-GAN":
            assert G is not None
            lr = LR(D, G, self.xin_t, self.num_classes, self.n_lr)
            train_stats = lr.fit()
            eval_stats = lr.eval(winv, woutv)
            self.lr_instance.append(lr)
            self.lr_train.append(train_stats)
            self.lr_overall.append(eval_stats)
        w_lst, legend_lst = [winv], ['InD']
        if each_class:
            assert cls_idx is not None
            for idx in cls_idx:
                ic(f"Class: {idx}")
                mask = yxoutv == idx
                woutv_idx = woutv[mask]
                tpr_95, tpr_95_thresh = tpr(winv, woutv_idx, 0.95)
                tpr_99, tpr_99_thresh = tpr(winv, woutv_idx, 0.99)
                result = [tpr_95, tpr_95_thresh, tpr_99, tpr_99_thresh]
                # plot
                w_lst.append(woutv_idx)
                legend_lst.append(f"OoD-[Class {idx}]")
                if self.method == 'OOD-GAN':
                    cls_lr_eval = lr.eval(winv, woutv_idx)
                    result += cls_lr_eval
                self.cls_stats[idx].append(result)
        else:
            w_lst.append(woutv)
            legend_lst.append(f"OoD")
        plot_wass_dist_and_thresh(
            w_lst, legend_lst, self.n_ood, self.log_dir, tag)

    def display_stats(self):
        # Overall stats
        print("\n" + line())
        print("Overall Statistics")
        print_stats(self.tpr95, "TPR@95TNR")
        print_stats(self.tpr95_thresh, "TPR@95TNR-Threshold")
        print_stats(self.tpr99, "TPR@99TNR")
        print_stats(self.tpr99_thresh, "TPR@99TNR-Threshold")
        print("\n" + line())
        if self.method == "OOD-GAN":
            lr_train = np.array(self.lr_train)
            print("Logistic Regression Statistics")
            print_stats(lr_train[:, 0], "Mean win")
            print_stats(lr_train[:, 1], "Mean wgz")
            print_stats(lr_train[:, 2], "Training Accuracy")
            lr_stats = np.array(self.lr_overall)
            print_stats(lr_stats[:, 0], "InD Accuracy")
            print_stats(lr_stats[:, 1], "OoD Accuracy")
            print_stats(lr_stats[:, 2], "AUROC")
        if len(self.cls_stats) != 0:
            for key, val in self.cls_stats.items():
                print("\n" + line())
                ic(f"Class: {key}")
                vals = np.array(val)
                print_stats(vals[:, 0], "TPR@95TNR")
                print_stats(vals[:, 1], "TPR@95TNR-Threshold")
                print_stats(vals[:, 2], "TPR@99TNR")
                print_stats(vals[:, 3], "TPR@99TNR-Threshold")
                if len(val) > 4:
                    lr_stats = vals[4:7]
                    print_stats(lr_stats[:, 0], "InD Accuracy")
                    print_stats(lr_stats[:, 1], "OoD Accuracy")
                    print_stats(lr_stats[:, 2], "AUROC")
