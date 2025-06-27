import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
from collections import defaultdict
from simulation import plot_heatmap_v2, DSIM, GSIM
import torch


parser = argparse.ArgumentParser(description="details")

# python get_results.py --name=FashionMNIST --regime=I
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--type', type=str, default="SEE-OOD")
parser.add_argument('--regime', type=str, default='I')

# Simulation results
# python get_results.py --sim --setting=I
# python get_results.py --sim --setting=II
parser.add_argument('--sim', action='store_true', default=False)
parser.add_argument('--setting', type=str, default=None)

# InD sensitivity analysis
# python get_results.py --ind_sa
parser.add_argument('--ind_sa', action='store_true', default=False)
args = parser.parse_args()


if args.sim:
    assert args.setting in ['I', 'II']
    setting = args.setting
    ckpt_dir = os.path.join('checkpoint', 'Simulation')
    IND_DATA, IND_X, IND_Y = torch.load(os.path.join(ckpt_dir, 'data', 'ind_data.pt'))
    OOD_DATA, OOD_X, OOD_Y = torch.load(os.path.join(ckpt_dir, 'data', 'ood_data.pt'))
    IND_DATA_TEST, IND_X_TEST, IND_Y_TEST = torch.load(os.path.join(ckpt_dir, 'data', 'ind_data_test.pt'))
    IND_CLS, OOD_CLS = torch.load(os.path.join(ckpt_dir, 'data', 'ind_ood_cls.pt'))
    OOD_BATCH = torch.load(os.path.join(ckpt_dir, 'data', 'OoDs' ,'OOD_2.pt'))
    # Load models
    D_GAN = DSIM(128)
    G_GAN = GSIM(128)
    D_WOOD = DSIM(128)
    D_GAN.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint', setting, 'D_GAN.pt')))
    G_GAN.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint', setting, 'G_GAN.pt')))
    D_WOOD.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint', setting, 'D_WOOD.pt')))
    # Start testing
    print(f"Testing and getting results for simulation setting {setting}...")
    save_dir = os.path.join('Document', 'Simulation')
    os.makedirs(save_dir, exist_ok=True)
    plot_heatmap_v2(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D_WOOD, None, 'WOOD', 
            IND_CLS, OOD_CLS, [0, 1, 2], [3], title="WOOD Wasserstein Score Heatmap",
            path=os.path.join(save_dir, 'WOOD.jpg'), tnr=0.95, lb=-1, ub=7, m=300)
    plot_heatmap_v2(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D_GAN, G_GAN, 'SEE-OOD', 
            IND_CLS, OOD_CLS, [0, 1, 2], [3], title=f"SEE-OoD Wasserstein Score Heatmap - Setting {setting}",
            path=os.path.join(save_dir, f'{setting}.jpg'), tnr=0.95, lb=-1, ub=7, m=300)
    exit()

if args.ind_sa:
    print(f"Compiling results for InD sample size sensitivity analysis...")
    N = [1000, 2000, 5000, 10000, 20000, 40000, 48000]
    labels = ["1K", '2K', '5K', '10K', '20K', '40K', '48K']
    AUCs, TPR95, TPR99, ACCs = [], [], [], []
    for n in N:
        file_path = os.path.join('checkpoint', 'log', 'FashionMNIST-InD-SA', f'log-64-{n}.txt')
        with open(file_path, 'r') as f:
                lines = f.readlines()
                # print(lines[index])
                ind_acc = float(lines[-20].split(" ")[-1].strip())
                tpr95 = float(lines[-13].split(" ")[1].strip())
                tpr99 = float(lines[-9].split(" ")[1].strip())
                auc = float(lines[-5].split(" ")[1].strip())
                # print(tpr95, tpr99, auc)
        AUCs.append(auc*100)
        TPR95.append(tpr95*100)
        TPR99.append(tpr99*100)
        ACCs.append(ind_acc*100)
    
    print(f"InD sample size sensitivity analysis results.")
    print(f"N: {N}")
    print(f"AUCs: {', '.join(f'{f*100:.4f}' for f in AUCs)}")
    print(f"ACCs: {', '.join(f'{f*100:.4f}' for f in ACCs)}")
    print(f"TPR95: {', '.join(f'{f*100:.4f}' for f in TPR95)}")
    print(f"TPR99: {', '.join(f'{f*100:.4f}' for f in TPR99)}\n\n")
    # plot
    mksz, lw = 6, 1.8
    N = np.arange(len(N))
    plt.plot(N, TPR95, label="TPR@95TNR", linestyle='solid', marker='s',  color="blue", linewidth=lw, markersize=mksz, alpha=1)
    plt.plot(N, TPR99, label="TPR@99TNR", linestyle='solid', marker='s', color='lightblue' ,linewidth=lw, markersize=mksz, alpha=1)
    plt.plot(N, AUCs, label="AUROC", linestyle='solid', marker='s',  color='red',linewidth=lw, markersize=mksz, alpha=1)
    plt.plot(N, ACCs, label="InD Accuracy", linestyle='-', marker='x', color='black',linewidth=lw, markersize=mksz, alpha=1)
    plt.legend(loc=4)
    plt.xticks(N, labels,fontdict={'fontsize': 9})
    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    plt.xlabel("Number of InD Training samples", fontdict={'fontsize': 14})
    plt.ylabel("%", fontdict={'fontsize': 13})
    plt.title(f"FashionMNIST sensitivity analysis on InD sample size", fontdict={'fontsize': 16})
    plt.savefig(f"Document/InD-SA/InD-SA.jpg", dpi=200)
    exit()





# FOR Baselines
save_dir = os.path.join('Document', args.name)
os.makedirs(save_dir, exist_ok=True)
# method_list = ['SEE-OOD', 'Energy-FT', 'OE', 'DeepSAD', 'ATD', 'Energy', 'VOS', 'ODIN', 'Maha']
# method_list = ['SEE-OOD', 'WOOD','Energy-FT', 'OE', 'ATD', 'Energy', 'VOS', 'ODIN', 'Maha']
method_list = ['SEE-OOD', 'WOOD', 'Energy-FT', 'OE']

N = np.arange(4, 13, 1)

# Result Dict:
RESULTS = {
    'TPR95': {},
    'TPR99': {},
    'AUC': {}
}

# File path:
LOGS = {
    'SEE-OOD': os.path.join('checkpoint', 'log', 'summary.txt'),
    'WOOD': os.path.join('..', 'WOOD', 'summary.txt'),
    'Energy-FT': os.path.join('..', 'energy_ood', 'CIFAR', 'out', 'energy_summary.txt'),
    'OE': os.path.join('..', 'energy_ood', 'CIFAR', 'out', 'oe_summary.txt'),
    'Energy': os.path.join('..', 'energy_ood', 'CIFAR', 'out', 'energy_baseline_summary.txt'),
    'DeepSAD': os.path.join('..', 'Deep-SAD-PyTorch', 'src', 'summary.txt'),
    'ATD': os.path.join('..', 'ATD', 'summary.txt'),
    'VOS': os.path.join('..', 'vos', 'classification', 'CIFAR', 'jobs', 'SEEOOD_baselines', 'summary.txt'),
    'ODIN': os.path.join('..', 'deep_Mahalanobis_detector', 'jobs', 'SEEOOD_Baselines', 'summary.txt'),
    'Maha': os.path.join('..', 'deep_Mahalanobis_detector', 'jobs', 'SEEOOD_Baselines', 'summary.txt'),
    'MSP': os.path.join('..', 'deep_Mahalanobis_detector', 'jobs', 'SEEOOD_Baselines', 'summary.txt') 
}

UNSUPERVISED_LIST = ['Energy', 'VOS', 'ODIN', 'Maha', 'MSP']

UNSUPERVISED_INDEX = {
    'MSP': {
    'MNIST': 2,
    'FashionMNIST': 7,
    'MNIST-FashionMNIST': 12,
    'SVHN': 17,
    'CIFAR10-SVHN': 22
    },
    'ODIN': {
    'MNIST': 33,
    'FashionMNIST': 38,
    'MNIST-FashionMNIST': 43,
    'SVHN': 48,
    'CIFAR10-SVHN': 53
    },
    'Maha': {
    'MNIST': 64,
    'FashionMNIST': 69,
    'MNIST-FashionMNIST': 74,
    'SVHN': 79,
    'CIFAR10-SVHN': 84
    },
    'VOS': {
    'MNIST': 0,
    'FashionMNIST': 6,
    'MNIST-FashionMNIST': 12,
    'SVHN': 18,
    'CIFAR10-SVHN': 24
    },
    'Energy': {
    'MNIST': 0,
    'FashionMNIST': 6,
    'MNIST-FashionMNIST': 12,
    'SVHN': 18,
    'CIFAR10-SVHN': 24
    }
}

INDEX = {
    'MNIST': 0,
    'FashionMNIST': 7,
    'MNIST-FashionMNIST': 14,
    'SVHN': 21,
    'CIFAR10-SVHN': 28,
    'FashionMNIST-R2': 35,
    'SVHN-R2': 42
}

for method in method_list:
    if method in UNSUPERVISED_LIST:
        with open(LOGS[method], 'r') as f:
            lines = f.readlines()
            target = UNSUPERVISED_INDEX[method][args.name]
            if method in ['VOS', 'Energy']:
                auc_list = [float(lines[target+1][4:].strip())] * len(N)
                tpr95_list = [float(lines[target+2][6:].strip())] * len(N)
                tpr99_list = [float(lines[target+3][6:].strip())] * len(N)
            else:
                tpr95_list = [float(lines[target+1][6:].strip())] * len(N)
                tpr99_list = [float(lines[target+2][6:].strip())] * len(N)
                auc_list = [float(lines[target+3][6:].strip())] * len(N)
    else:
        with open(LOGS[method], 'r') as f:
            lines = f.readlines()
            target = INDEX[args.name]
            auc_list = np.array(lines[target + 2][6:].strip().split(", "), dtype=np.float32)
            tpr95_list = np.array(lines[target + 3][6:].strip().split(", "), dtype=np.float32)
            tpr99_list = np.array(lines[target + 4][6:].strip().split(", "), dtype=np.float32)
            # Append results
    RESULTS['TPR95'][method] = tpr95_list
    RESULTS['TPR99'][method] = tpr99_list
    RESULTS['AUC'][method] = auc_list

# print(RESULTS)

# Plot TPR95
# Config
mksz = 6
lw = 1.8
fontsize=10

markers = ['s', 'x', 'o', 'v', '^', '>', '<', '+', 'D', 'p', '*', 'h', '|']
NEXP = [r'$\mathregular{2^4}$',
        r'$\mathregular{2^5}$',
        r'$\mathregular{2^6}$',
        r'$\mathregular{2^7}$',
        r'$\mathregular{2^8}$',
        r'$\mathregular{2^9}$',
        r'$\mathregular{2^{10}}$',
        r'$\mathregular{2^{11}}$',
        r'$\mathregular{2^{12}}$']

for i, method in enumerate(method_list):
    print(RESULTS['TPR95'][method][1:])
    if method in UNSUPERVISED_LIST:
        plt.plot(N, RESULTS['TPR95'][method], label=method, linestyle='solid', marker=markers[i], linewidth=lw, markersize=mksz, alpha=1)
    else:
        plt.plot(N, RESULTS['TPR95'][method][1:], label=method, linestyle='solid', marker=markers[i], linewidth=lw, markersize=mksz, alpha=1)

plt.legend(loc=4)
plt.xticks(N, NEXP, fontdict={'fontsize': 12})
if args.name not in ['FashionMNIST-R2', 'SVHN-R2']:
    plt.xlabel("Number of OoD samples for EACH class", fontdict={'fontsize': 14})
else:
    plt.xlabel("Number of OoD samples for SELECTED class", fontdict={'fontsize': 14})
plt.ylabel("TPR (%)", fontdict={'fontsize': 13})
plt.title(f"Regime {args.regime} - {args.name} - TPR at 95 TNR", fontdict={'fontsize': 16})
plt.savefig(f"{save_dir}/{args.name}-Regime-{args.regime}-95.jpg", dpi=500)
# plt.show()
plt.close()

# TPR99
for i, method in enumerate(method_list):
    if method in UNSUPERVISED_LIST:
        plt.plot(N, RESULTS['TPR99'][method], label=method, linestyle='solid', marker=markers[i], linewidth=lw, markersize=mksz, alpha=1)
    else:
        plt.plot(N, RESULTS['TPR99'][method][1:], label=method, linestyle='solid', marker=markers[i], linewidth=lw, markersize=mksz, alpha=1)

plt.legend(loc=4)
plt.xticks(N, NEXP, fontdict={'fontsize': 12})
if args.name not in ['FashionMNIST-R2', 'SVHN-R2']:
    plt.xlabel("Number of OoD samples for EACH class", fontdict={'fontsize': 14})
else:
    plt.xlabel("Number of OoD samples for SELECTED class", fontdict={'fontsize': 14})
plt.ylabel("TPR (%)", fontdict={'fontsize': 13})
plt.title(f"Regime {args.regime} - {args.name} - TPR at 99 TNR", fontdict={'fontsize': 16})
plt.savefig(f"{save_dir}/{args.name}-Regime-{args.regime}-99.jpg", dpi=500)
# plt.show()
plt.close()