from ood_gan import *
from models.dc_gan_model import *
from dataset import *
from config import *
from models.model import *
from eval import *
import argparse
import time
import yaml
from scipy import stats

# torch.manual_seed(24)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset')
parser.add_argument('--regime', help="Experimental Regime", type=str)
args = parser.parse_args()


# dset='CIFAR10-SVHN'
if args.dataset == 'CIFAR10-SVHN':
    dset = DSET('CIFAR10-SVHN', False, 100, 100, None, None)
    n_cls=10
    C=3
elif args.dataset == 'CIFAR10-Texture':
    dset = DSET('CIFAR10-Texture', False, 100, 100, None, None)
    n_cls=10
    C=3
elif args.dataset == 'SVHN':
    dset = DSET('SVHN', True, 128, 128, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    n_cls=8
    C=3
elif args.dataset == 'FashionMNIST':
    dset = DSET('FashionMNIST', True, 100, 100, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    n_cls=8
    C=1
    print('Hello FashionMNIST')
elif args.dataset == 'MNIST-FashionMNIST':
    dset = DSET('MNIST-FashionMNIST', False, 100, 100, None, None)
    n_cls=10
    C=1

# N = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# MC = [0, 1, 2]

N = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
MC = [0, 1, 2]
EPOCH = 15

AUROCS = []
AUROCS_MAD = []
TPR95 = []
TPR99 = []
TPR95_MAD = []
TPR99_MAD = []


for n in N:
    auroc_n, tpr95_n, tpr99_n = [], [], []
    # for mc in MC:
    # ckpt_name = f"[{dset.name}]-[{n}]-[{args.regime}]-[{mc}]_[{EPOCH}].pt"
    # ckpt_path = f"/home/xysong/Out-of-Distribution-GANs/checkpoint/OOD-GAN/{dset.name}/{args.regime}/{n}/{ckpt_name}"

    ckpt_path = f"/home/xysong/Out-of-Distribution-GANs/checkpoint/OOD-GAN/{dset.name}/{args.regime}/{n}/eval.pt"
    # ckpt_path = f"/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/OOD-GAN/{dset.name}/{args.regime}/{n}/eval.pt"
        
        # if args.regime == 'Imbalanced':
        #     ckpt_path = f"/home/xysong/WOOD/runs/{dset.name}-R2/{n}/OOD_model_{mc}.t7"
        # else:
        #     # ckpt_path = f"/home/xysong/WOOD/runs/{dset.name}/{n}/OOD_model_{mc}.t7"

        #     # For texture
        #     ckpt_path = f"/scratch/sunwbgt_root/sunwbgt98/xysong/WOOD/runs/{dset.name}/{n}/OOD_model_{mc}.t7"
        
    print(ckpt_path)
    ckpt = torch.load(ckpt_path)

        # D = DenseNet3(depth=100, num_classes=n_cls, input_channel=C).to(DEVICE)
        # D.load_state_dict(ckpt['D-state'])
        # D = nn.DataParallel(D)
        # D.load_state_dict(ckpt)
        # D.eval()

    winv, woutv = ckpt.winv, ckpt.woutv
    # print(len(winv))
    with torch.no_grad():
        for mc in MC:
            auroc = naive_auroc(-winv[mc].cpu(), -woutv[mc].cpu())
            tpr95, _ = tpr(winv[mc].cpu(), woutv[mc].cpu(), 0.95)
            tpr99, _ = tpr(winv[mc].cpu(), woutv[mc].cpu(), 0.99)
            auroc_n.append(auroc)
            tpr95_n.append(tpr95)
            tpr99_n.append(tpr99)


            # tpr95, tpr99, auroc = evaluate(D, dset.ind_val_loader, dset.ood_val_loader)
            # auroc_n.append(auroc)
            # tpr95_n.append(tpr95)
            # tpr99_n.append(tpr99)

    # logging
    print(f"N = {n}: {np.mean(tpr95_n):.4f} - {np.mean(tpr99_n):.4f} | {np.mean(auroc_n):.4f}")
    AUROCS.append(np.mean(auroc_n))
    AUROCS_MAD.append(stats.median_abs_deviation(auroc_n, axis=None))
    TPR95.append(np.mean(tpr95_n))
    TPR95_MAD.append(stats.median_abs_deviation(tpr95_n, axis=None))
    TPR99.append(np.mean(tpr99_n))
    TPR99_MAD.append(stats.median_abs_deviation(tpr99_n, axis=None))


print('+'* 80)
print(f"Results - {args.dataset}")
print("TPR95" + "=" * 80)
print(np.round(TPR95, 4))
print(np.round(TPR95_MAD, 4))
print("TPR99" + "=" * 80)
print(np.round(TPR99, 4))
print(np.round(TPR99_MAD, 4))
print("AUROC" + "=" * 80)
print(np.round(AUROCS, 4))
print(np.round(AUROCS_MAD, 4))

