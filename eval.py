import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from tqdm import tqdm
from collections import defaultdict
from config import *
from dataset import *
from models.gans import *
from models.dc_gan_model import *
from models.model import *
from models.ood_gan_backbone import *
from ood_gan import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def naive_auroc(winv, woutv):
    pos = np.array(winv[:]).reshape((-1, 1))
    neg = np.array(woutv[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = roc_auc_score(labels, examples)

    return auroc

def auroc_log(winv, woutv):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    # tpr 99
    tnr_at_tpr99 = dict()
    known, novel = winv, woutv
    known.sort()
    novel.sort()
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp= -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n

    # TPR 99
    tpr99_pos = np.abs(tp/ num_k - .99).argmin()
    tnr_at_tpr99 = 1. - fp[tpr99_pos] / num_n
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    auroc = -np.trapz(1.-fpr, tpr)
    return auroc, tnr_at_tpr95, tnr_at_tpr99

def tpr(winv, woutv, level=0.95):
    assert level < 1 and level > 0
    threshold = np.quantile(winv.to('cpu'), level)
    # print(f"{level*100}% TNR Threshold: {threshold}")
    fpr = woutv[woutv <= threshold].shape[0] / float(woutv.shape[0])
    tpr = 1 - fpr
    # print(f"TPR at {level*100}%  TNR: {tpr}")
    return tpr, threshold


def plot_wass_dist_and_thresh(wass_lst, legend_lst, n_ood, log_dir, tag,
                              thresh_lst=None, thresh_lbl_lst=None, bins=50, alpha=0.5):
    assert len(wass_lst) == len(legend_lst)
    for wass, lbl in zip(wass_lst, legend_lst):
        plt.hist(wass.cpu().numpy(), bins=bins, alpha=alpha, label=lbl)
    # for thresh, thresh_lbl in zip(thresh_lst, thresh_lbl_lst):
    #     plt.axvline(x=thresh, color='b', label=thresh_lbl)
    plt.legend()
    plt.xlabel("Wasserstein Distance")
    plt.ylabel("Count")
    plt.title(f"Number of Observed OoD Samples: {n_ood}")
    plt.savefig(log_dir + f"wass_plot[{n_ood}]-[{tag}].jpg", dpi=1000)
    plt.close()


def loader_wass(data_loader, D, loss_type='Dynamic_Wasserstein'):
    with torch.no_grad():
        wass_dists = []
        # ic(DEVICE)
        # assert DEVICE == 'cuda'
        for (img, label) in tqdm(data_loader):
            label = torch.tensor(label, dtype=torch.int64).to(DEVICE)
            img = img.to(DEVICE)
            out = D(img)
            # wass_dist = ood_wass_loss_dynamic(torch.softmax(out, dim=-1))
            if loss_type == 'Dynamic_Wasserstein':
                wass_dist = sink_dist_test(torch.softmax(out, dim=-1), label, out.shape[1]).cpu().detach()
            else:
                # wass_dist = ood_wass_loss(torch.softmax(out, dim=-1))
                wass_dist = 1 - torch.max(torch.softmax(out, dim=-1), dim=-1)[0]
            # wass_dist = sink_dist_test_v2(torch.softmax(out, dim=-1), None, 8)
            # img = img.to('cpu')
            # wass_dist = ood_wass_loss(out)
            wass_dists.append(wass_dist)
        return torch.cat(wass_dists, dim=0)


def print_stats(stat, name, precision=5):
    print(f"{name}: {stat}")
    mad = np.mean(np.abs(np.mean(stat) - stat))
    print(
        f"mean: {np.round(np.mean(stat), precision)} | std: {np.round(np.std(stat), precision)} | MAD: {np.round(mad, precision)}")


def ic_stats(stat, precision=5):
    ic(stat)
    ic(f"mean: {np.round(np.mean(stat), precision)} | std: {np.round(np.std(stat), precision)}")


class EVALER():
    def __init__(self, xin_t, xin_v, xin_v_loader, xout_v, xout_v_loader,
                 n_ood, log_dir, method, num_classes, n_lr=2000, loss_type='Dynamic_Wasserstein'):
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
        self.tpr95_raw, self.tpr99_raw = [], []
        self.tpr95_thresh, self.tpr99_thresh = [], []
        self.auroc = []
        self.loss_type = loss_type
        # Statistics - Logistic Regression
        # Refer to the return values of logistic regression for details.
        self.lr_instance = []
        self.lr_train = []
        self.lr_overall = []
        self.lrtpr, self.tpr_lr_thresh = [], []
        # Refer to class_level statistics
        self.cls_stats = defaultdict(list)

    def save(self, path):
        # Do not save unnecessary stuffs
        self.xin_t, self.xin_v = None, None  # InD Testing dataset & loader
        self.xin_v_loader = None
        self.xout_v = None  # OoD Testing dataset & loader
        self.xout_v_loader = None
        torch.save(self, path)

    def evaluate(self, D, tag, G=None, each_class=False, cls_idx=None):
        print("Computing evaluation statistics...")
        _, yxoutv = tuple_list_to_tensor(self.xout_v)
        print("> Evaluating InD Wasserstein distances...")
        winv = loader_wass(self.xin_v_loader, D, self.loss_type)
        print("> Evaluating OoD Wasserstein distances...")
        woutv = loader_wass(self.xout_v_loader, D, self.loss_type)
        self.winv.append(winv)
        self.woutv.append(woutv)
        # Test model performance
        tpr_95, tpr_95_thresh = tpr(winv, woutv, 0.95)
        tpr_99, tpr_99_thresh = tpr(winv, woutv, 0.99)
        self.tpr95.append(tpr_95)
        self.tpr95_thresh.append(tpr_95_thresh)
        self.tpr99.append(tpr_99)
        self.tpr99_thresh.append(tpr_99_thresh)
        auc = naive_auroc(-winv.cpu(), -woutv.cpu())
        self.auroc.append(auc)
        # auroc, tnr_at_tpr95, tnr_at_tpr99 = auroc_log(winv, woutv)
        # self.auroc.append(auroc)
        # self.tpr95_raw.append(tnr_at_tpr95)
        # self.tpr99_raw.append(tnr_at_tpr99)
        # yxoutv = None

        # w_lst, legend_lst = [winv], ['InD']
        # if each_class:
        #     assert cls_idx is not None
        #     for idx in cls_idx:
        #         print(f"Class: {idx}")
        #         mask = yxoutv == idx
        #         woutv_idx = woutv[mask]
        #         tpr_95, tpr_95_thresh = tpr(winv, woutv_idx, 0.95)
        #         tpr_99, tpr_99_thresh = tpr(winv, woutv_idx, 0.99)
        #         result = [tpr_95, tpr_95_thresh, tpr_99, tpr_99_thresh]
        #         # plot
        #         w_lst.append(woutv_idx)
        #         legend_lst.append(f"OoD-[Class {idx}]")
        #         self.cls_stats[idx].append(result)
        # else:
        #     w_lst.append(woutv)
        #     legend_lst.append(f"OoD")
        # plot_wass_dist_and_thresh(
        #     w_lst, legend_lst, self.n_ood, self.log_dir, tag)

    def display_stats(self):
        # Overall stats
        print("\n" + line())
        print("Overall Statistics")
        print_stats(self.tpr95, "TPR@95TNR")
        print_stats(self.tpr95_thresh, "TPR@95TNR-Threshold")
        print_stats(self.tpr99, "TPR@99TNR")
        print_stats(self.tpr99_thresh, "TPR@99TNR-Threshold")
        print_stats(self.auroc, "AUROC")
        # print_stats(self.auroc, "AUROC")
        # print_stats(self.tpr95_raw, "TPR@95RAW")
        # print_stats(self.tpr99_raw, "TPR@99RAW")
        print("\n" + line())


def evaluate(D, ind_val, ood_val, loss_type='Dynamic_Wasserstein'):
        print("Computing evaluation statistics...")
        print("> Evaluating InD Wasserstein distances...")
        winv = loader_wass(ind_val, D, loss_type)
        print("> Evaluating OoD Wasserstein distances...")
        woutv = loader_wass(ood_val, D, loss_type)
        # print(len(winv))
        # print(winv[0:10])
        # print(len(woutv))
        # print(woutv[0:10])
        # Test model performance
        tpr_95, tpr_95_thresh = tpr(winv, woutv, 0.95)
        tpr_99, tpr_99_thresh = tpr(winv, woutv, 0.99)
        # auroc, tnr_at_tpr95, tnr_at_tpr99 = auroc_log(winv, woutv)
        print("\nOverall Statistics")
        print_stats(tpr_95, "TPR@95TNR")
        print_stats(tpr_95_thresh, "TPR@95TNR-Threshold")
        print_stats(tpr_99, "TPR@99TNR")
        print_stats(tpr_99_thresh, "TPR@99TNR-Threshold\n")
        # print_stats(auroc, "AUROC")
        # print_stats(tnr_at_tpr95, "TPR@95RAW")
        # print_stats(tnr_at_tpr99, "TPR@99RAW")
        auc = naive_auroc(-winv.cpu(), -woutv.cpu())
        print(f"AUROC: {auc}\n")
        return tpr_95, tpr_99, auc


def plot_loss_curve(d_loss, g_loss, path):
    ce, w_ood, w_d = d_loss[:,0], d_loss[:,1], d_loss[:,2]
    w_g = g_loss
    iters = range(0, len(d_loss), 1)
    fig, axs = plt.subplots(2, sharex=True)
    fig.tight_layout(pad=2.0)
    # fig.suptitle('Training Loss Curves')
    # Discriminator Loss
    axs[0].plot(iters, ce, label='CE', marker='^', markersize=3)
    axs[0].plot(iters, w_ood, label=r'$W_{OoD}$', marker='o', markersize=3)
    axs[0].plot(iters, w_d, label=r'$W_{Z}$', marker='x', markersize=3)
    # axs[0].set_xlabel('Training Epochs')
    axs[0].set_ylabel('Loss Value')
    axs[0].set_title("Discriminator Loss Curve")
    axs[0].legend()
    # Generator Loss
    axs[1].plot(iters, w_g, label=r'$W_{Z}$', marker='o', markersize=3)
    # axs[1].set_xlabel('Training Epochs')
    axs[1].set_ylabel('Loss Value')
    axs[1].set_title("Generator Loss Curve")
    axs[1].legend()
    fig.savefig(path, dpi=1500)
    plt.close()

def plot_loss_curve_from_log(filename, length, path, both=True):
    with open(filename, 'r') as f:
        lines = f.readlines()
        n=0
        dloss, gloss = [], []
        for line in lines:
            if "Step" in line:
                ce, w_ood, w_z = float(line[31:37]), float(line[48:54]), float(line[63:69])
                g_w_z = float(line[91:97])
                dloss.append([ce, w_ood, w_z])
                gloss.append(g_w_z)
                # print(ce, w_ood, w_z, g_w_z)
                n += 1
            
            if n >= length: break
        
        dloss = np.array(dloss)
        gloss = np.array(gloss)
        
        # plot
        ce, w_ood, w_d = dloss[:,0], dloss[:,1], dloss[:,2]
        w_g = gloss
        iters = np.arange(0, n, 1) * 20
        if both:
            fig, axs = plt.subplots(2, sharex=True)
            fig.tight_layout(pad=2.0)
            # fig.suptitle('Training Loss Curves')
            # Discriminator Loss
            axs[0].plot(iters, ce, label='CE', marker='^', markersize=3)
            axs[0].plot(iters, w_ood, label=r'$S_{OoD}$', marker='o', markersize=3)
            axs[0].plot(iters, w_d, label=r'$S_{G(Z)}$', marker='x', markersize=3)
            # axs[0].set_xlabel('Training Epochs')
            axs[0].set_yticks([0, 1, 2])
            axs[0].set_ylabel('Value of Key Terms')
            axs[0].set_title("Discriminator")
            axs[0].legend()
            axs[0].grid(axis='y', linestyle='--', alpha=0.7) 
            # Generator Loss
            axs[1].plot(iters, w_g, label=r'$S_{G(Z)}$', marker='x', markersize=3, color="green")
            axs[1].set_xlabel('Number of updates')
            axs[1].set_yticks([0, 1, 2])
            axs[1].set_ylabel('Value of Key Terms')
            axs[1].set_title("Generator")
            axs[1].legend()
            axs[1].grid(axis='y', linestyle='--', alpha=0.7) 
            fig.savefig(path, dpi=1500)
            plt.close()
        else:
            plt.plot(iters, ce, label='CE', marker='^', markersize=3)
            plt.plot(iters, w_ood, label=r'$S_{OoD}$', marker='o', markersize=3)
            plt.plot(iters, w_d, label=r'$S_{G(Z)}$', marker='x', markersize=3)
            plt.ylabel("Values")
            plt.xlabel("Number of iterations")
            plt.title("Trajectory of Key Terms")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7) 
            plt.savefig(path, dpi=200)
            plt.close()

def plot_loss_curve_from_log_hparam(dir_name, length, path, beta_ood, beta_z):
    filename = os.path.join(dir_name, f"log-{beta_ood}-{beta_z}.txt")
    with open(filename, 'r') as f:
        lines = f.readlines()
        n=0
        dloss, gloss = [], []
        for line in lines:
            if "Step" in line:
                ce, w_ood, w_z = float(line[31:37]), float(line[48:54]), float(line[63:69])
                g_w_z = float(line[91:97])
                dloss.append([ce, w_ood, w_z])
                gloss.append(g_w_z)
                n += 1
            
            if n >= length: break
        
        dloss = np.array(dloss)
        gloss = np.array(gloss)
        
        # plot
        ce, w_ood, w_d = dloss[:,0], dloss[:,1], dloss[:,2]
        w_g = gloss
        iters = np.arange(0, n, 1) * 20

        plt.plot(iters, ce, label='CE', marker='^', markersize=3)
        plt.plot(iters, w_ood, label=r'$S_{OoD}$', marker='o', markersize=3)
        plt.plot(iters, w_d, label=r'$S_{G(Z)}$', marker='x', markersize=3)
        plt.ylabel("Values")
        plt.xlabel("Number of iterations")
        plt.title(rf"Trajectory of Key Terms ($\beta_{{OoD}}$={beta_ood}, $\beta_{{z}}$={beta_z})")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7) 
        os.makedirs(path, exist_ok=True)
        savepath = os.path.join(path, f"{beta_ood}-{beta_z}.png")
        plt.savefig(savepath, dpi=200)
        plt.close()


if __name__ == "__main__":

    pass
    n=150
    both=False
    # filename = os.path.join('checkpoint', 'log', 'FashionMNIST', 'log-64.txt')
    # plot_loss_curve_from_log(filename, n, 'Document/Loss/FashionMNIST-64-0.png', both)

    # filename = os.path.join('checkpoint', 'log', 'SVHN', 'log-32.txt')
    # plot_loss_curve_from_log(filename, 1000, 'Document/Loss/SVHN-32.png')

    # filename = os.path.join('checkpoint', 'log', '3DPC-R2', 'log-200.txt')
    # plot_loss_curve_from_log(filename, n, 'Document/Loss/3DPC-R2-200-0.png', both)

    # filename = os.path.join('checkpoint', 'log', 'CIFAR10-SVHN', 'log-32.txt')
    # plot_loss_curve_from_log(filename, 1000, 'Document/Loss/CIFAR10-SVHN-32.png')

    # filename = os.path.join('checkpoint', 'log', 'MNIST', 'log-32.txt')
    # plot_loss_curve_from_log(filename, 1000, 'Document/Loss/MNIST-32.png')

    # filename = os.path.join('checkpoint', 'log', 'other', 'bad-convergence-2.txt')
    # plot_loss_curve_from_log(filename, 1000, 'Document/Loss/FashionMNIST-64-bad.png')

    # filename = os.path.join('checkpoint', 'log', 'other', 'bad-convergence-3.txt')
    # plot_loss_curve_from_log(filename, n, 'Document/Loss/FashionMNIST-64-bad-3-0.png', both)

    for beta_ood in [0.001, 0.01, 0.1, 1, 10]:
        for beta_z in [0.001, 0.01, 0.1, 1, 10]:
            dir_name = os.path.join('checkpoint', 'log', 'Param-SA')
            save_dir = os.path.join('Document', 'Loss', 'Param-SA')
            plot_loss_curve_from_log_hparam(dir_name, n, save_dir, beta_ood, beta_z)
