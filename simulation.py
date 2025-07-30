import torch
import numpy as np
from numpy.random import multivariate_normal as mn
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from config import DEVICE
from itertools import product
from wasserstein import batch_wasserstein, ood_wass_loss, batch_wasserstein_og
# For argument processing
import argparse
import time
import yaml
import os

class N():
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
    def sample(self, n):
        return mn(self.mu, self.cov, n)

# Single layer NN
class GSIM_SINGLE(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.fc1 = nn.Linear(2, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, 2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, z):
        return self.fc2(self.relu(self.fc1(z)))
        
class DSIM_SINGLE(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.fc1 = nn.Linear(2, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, 3)
        # self.fc = nn.Linear(2, 3)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out
    
# Two layer NN
class GSIM(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.fc1 = nn.Linear(2, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 2)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, z):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(z)))))

class DSIM(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.fc1 = nn.Linear(2, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 3)
        # self.fc = nn.Linear(2, 3)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        # return self.fc(x)
    
def get_sampler(mu, cov, n):
    SAMPLERS = {}
    for idx in range(n):
        SAMPLERS[idx] = N(mu[idx], cov[idx])
    return SAMPLERS

def get_train_test_samples(SAMPLERS, n):
    X_TRAIN, Y_TRAIN = {}, {}
    X_TEST, Y_TEST = {}, {}

    for cls in SAMPLERS:
        X_TRAIN[cls] = SAMPLERS[cls].sample(n)
        Y_TRAIN[cls] = np.array([cls]*n)
        X_TEST[cls] = SAMPLERS[cls].sample(n)
        Y_TEST[cls] = np.array([cls]*n)
    return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST

def cls_to_dset(idxs, X, Y):
    x, y = [], []
    for idx in idxs:
        x.extend(X[idx])
        y.extend(Y[idx])
    x = np.array(x)
    y = np.array(y)
    return list(zip(x, y)), x, y

def classifier_training(D, criterion, optimizer, ind_tri_loader, ind_val_loader, max_epoch, n_epoch=10):
    # Simple training loop
    iter_count_train = 0
    iter_count_val = 0
    for epoch in tqdm(range(max_epoch)):
        # Training
        D.train()
        train_loss, train_acc, wass = [], [], []
        for idx, (img, labels) in enumerate(ind_tri_loader):
            img = img.to(torch.float32)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = D(img)
            # print(labels)
            loss = criterion(logits, labels)
            loss.backward()
            # ic(loss)
            optimizer.step()
            # Append training statistics
            acc = (torch.argmax(logits, dim=1) ==
                labels).sum().item() / labels.shape[0]
            train_acc.append(acc)
            train_loss.append(loss.detach().item())
            iter_count_train += 1
        if epoch % n_epoch == 0:
            print(f"Epoch  # {epoch + 1} | Tri loss: {np.round(np.mean(train_loss), 4)} \
                    | Tri accuracy: {np.round(np.mean(train_acc), 4)}")
        # Evaluation
        D.eval()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for idx, (img, labels) in enumerate(ind_val_loader):
                img, labels = img.to(torch.float32), labels.to(DEVICE)
                logits = D(img)
                loss = criterion(logits, labels)
                acc = (torch.argmax(logits, dim=1) ==
                    labels).sum().item() / labels.shape[0]
                val_acc.append(acc)
                val_loss.append(loss.detach().item())
                iter_count_val += 1
            if epoch % n_epoch == 0:
                print(f"Epoch  # {epoch + 1} | Val loss: {np.round(np.mean(val_loss), 4)} \
                    | Val accuracy: {np.round(np.mean(val_acc), 4)}")
    return D

def wood_training(D, OOD_BATCH, ood_bsz, beta, criterion, optimizer, ind_tri_loader, ind_val_loader, max_epoch, n_epoch=10, f=None):
    # Simple training loop
    ood_batch = OOD_BATCH.to(DEVICE)
    iter_count_train, iter_count_val = 0, 0
    for epoch in tqdm(range(max_epoch)):
        # Training
        D.train()
        train_loss, train_acc, wass = [], [], []
        for idx, (img, labels) in enumerate(ind_tri_loader):
            img = img.to(torch.float32).to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = D(img)
            # print(labels)
            ood_idx = np.random.choice(
            len(ood_batch), min(len(ood_batch), ood_bsz), replace=False)
            ood_samples = ood_batch[ood_idx, :].to(DEVICE)
            ood_logits = D(ood_samples)
            # print(torch.softmax(ood_logits, dim=-1))
            # ic(ood_logits.shape)
            wass_loss = batch_wasserstein_og(ood_logits)
            loss = criterion(logits, labels) - beta * wass_loss
            # loss = criterion(logits, labels)
            loss.backward()
            # ic(loss)
            optimizer.step()
            # Append training statistics
            acc = (torch.argmax(logits, dim=1) ==
                labels).sum().item() / labels.shape[0]
            train_acc.append(acc)
            train_loss.append(loss.detach().item())
            iter_count_train += 1

        if epoch % n_epoch == 0:
            print(f"Epoch  # {epoch + 1} | Tri loss: {np.round(np.mean(train_loss), 4)} \
                    | Tri accuracy: {np.round(np.mean(train_acc), 4)}")
            print(f"Wasserstein Scores: {wass_loss}")
            if f is not None:
                f.write(f"Epoch  # {epoch + 1} | Tri loss: {np.round(np.mean(train_loss), 4)} \
                    | Tri accuracy: {np.round(np.mean(train_acc), 4)}\n")
                f.write(f"Wasserstein Scores: {wass_loss}\n")
        # Evaluation
        D.eval()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for idx, (img, labels) in enumerate(ind_val_loader):
                img, labels = img.to(torch.float32).to(DEVICE), labels.to(DEVICE)
                logits = D(img)
                loss = criterion(logits, labels)
                acc = (torch.argmax(logits, dim=1) ==
                    labels).sum().item() / labels.shape[0]
                val_acc.append(acc)
                val_loss.append(loss.detach().item())
                iter_count_val += 1
            if epoch % n_epoch == 0:
                print(f"Epoch  # {epoch + 1} | Val loss: {np.round(np.mean(val_loss), 4)} \
                    | Val accuracy: {np.round(np.mean(val_acc), 4)}")
                if f is not None:
                    f.write(f"Epoch  # {epoch + 1} | Val loss: {np.round(np.mean(val_loss), 4)} \
                    | Val accuracy: {np.round(np.mean(val_acc), 4)}\n")
    return D

def oodgan_training(D, G, D_solver, G_solver, OOD_BATCH, ood_bsz, bsz_tri, w_ce, w_wass_ood, w_wass_gz, w_dist, \
                    d_step_ratio, g_step_ratio, ind_tri_loader, ind_val_loader, max_epoch, n_epoch=10, n_step_log = 100, f=None):
    
    assert d_step_ratio == 1 or g_step_ratio == 1

    def ood_gan_d_loss(logits_real, logits_fake, logits_ood, labels_real):
        # 1: CrossEntropy of X_in
        criterion = nn.CrossEntropyLoss()
        ind_ce_loss = criterion(logits_real, labels_real)
        # 2. W_ood
        assert logits_ood.requires_grad
        w_ood = batch_wasserstein_og(logits_ood)
        # 3. W_z
        assert logits_fake.requires_grad
        w_fake = batch_wasserstein_og(logits_fake)
        return ind_ce_loss, w_ood, w_fake
    
    def ood_gan_g_loss(logits_fake, gz, xood):
        # 1. Wasserstein distance of G(z)
        assert logits_fake.requires_grad
        w_fake = batch_wasserstein_og(logits_fake)
        # distance term
        # print(torch.mean(gz, dim=0) - torch.mean(xood, dim=0))
        dist = torch.sqrt(torch.sum((torch.mean(gz) - torch.mean(xood))**2))
        assert dist.requires_grad
        return w_fake, dist
    
    iter_count = 0
    ood_batch = OOD_BATCH.to(DEVICE)
    # Logging
    D_loss, G_loss = [], []
    trajectory = []
    for epoch in tqdm(range(max_epoch)):
        D.train()
        G.train()
        for steps, (x, y) in enumerate(ind_tri_loader):
            x,y = x.to(torch.float32).to(DEVICE), y.to(DEVICE)
            # ---------------------- #
            # DISCRIMINATOR TRAINING #
            # ---------------------- #
            for d_step in range(d_step_ratio):
                D_solver.zero_grad()
                # Logits for X_in
                logits_real = D(x)

                seed = torch.rand((bsz_tri, 2), device=DEVICE)
                # Gz = self.G(seed, [cls]*self.bsz_tri).detach()

                Gz = G(seed)

                # Gz = self.G(seed).to(DEVICE).detach()
                logits_fake = D(Gz)
                # Logits for X_ood
                ood_idx = np.random.choice(len(ood_batch), min(
                    len(ood_batch), ood_bsz), replace=False)
                ood_sample = ood_batch[ood_idx, :].to(DEVICE)
                logits_ood = D(ood_sample)

                # Compute loss
                ind_ce_loss, w_ood, w_fake = ood_gan_d_loss(
                    logits_real, logits_fake, logits_ood, y)
                # print(w_ood)
                d_total = w_ce * ind_ce_loss -w_wass_ood * w_ood + w_fake * w_wass_gz
                # Update
                d_total.backward()
                D_solver.step()

            # ------------------ #
            # GENERATOR TRAINING #
            # ------------------ #
            for g_step in range(g_step_ratio):
                seed = torch.rand((bsz_tri, 2), device=DEVICE)
                # Gz = self.G(seed, [cls]*self.bsz_tri).detach()

                Gz = G(seed)
                G_solver.zero_grad()
                logits_fake = D(Gz)
                w_z, dist = ood_gan_g_loss(logits_fake, Gz, ood_sample)
                # g_total = -w_wass * (w_z) + dist * w_dist
                g_total = -w_wass_gz * w_z
                # Update
                g_total.backward()
                G_solver.step()

            # Print out statistics
            if (iter_count % n_step_log == 0):
                print(
                    f"Step: {steps:<4} | D: {d_total.item(): .4f} | CE: {ind_ce_loss.item(): .4f} | W_OoD: {w_ood.item(): .4f} | W_z: {w_fake.item(): .4f} | G: {g_total.item(): .4f} | W_z: {w_z.item(): .4f} | dist: {dist:.4f}")
                if f is not None:
                    f.write(
                    f"Step: {steps:<4} | D: {d_total.item(): .4f} | CE: {ind_ce_loss.item(): .4f} | W_OoD: {w_ood.item(): .4f} | W_z: {w_fake.item(): .4f} | G: {g_total.item(): .4f} | W_z: {w_z.item(): .4f} | dist: {dist:.4f}\n")
            iter_count += 1
        
        D_loss.append((ind_ce_loss.detach().item(), w_ood.detach().item(), w_fake.detach().item()))
        G_loss.append((w_z.detach().item(), dist.detach().item()))
        trajectory.append(np.array(torch.mean(Gz.detach(), dim=0).cpu()))

        D.eval()
        with torch.no_grad():
            val_acc = []
            for idx, (img, labels) in enumerate(ind_val_loader):
                img, labels = img.to(torch.float32).to(DEVICE), labels.to(DEVICE)
                logits = D(img)
                acc = (torch.argmax(logits, dim=1) ==
                    labels).sum().item() / labels.shape[0]
                val_acc.append(acc)
            if epoch % n_epoch == 0:
                print(f"Epoch  # {epoch + 1} | Val accuracy: {np.round(np.mean(val_acc), 4)}")
                if f is not None:
                    f.write(f"Epoch  # {epoch + 1} | Val accuracy: {np.round(np.mean(val_acc), 4)}\n")
    
    return D, G, (D_loss, G_loss, np.array(trajectory))

def calculate_accuracy(D, ind, ood, tnr, simplified=False):
    z = torch.softmax(D(torch.tensor(ind, dtype=torch.float32).to(DEVICE)), dim=-1)
    # print(z.shape)
    if simplified:
        s = 1 - torch.max(z, dim=-1)[0]
    else:
        s = ood_wass_loss(z)

    # print(s.shape)
    threshold = np.quantile(s.cpu().detach().numpy(), tnr)
    # print(threshold)
    z_ood = torch.softmax(D(torch.tensor(ood, dtype=torch.float32).to(DEVICE)), dim=-1)
    # print(z_ood.shape)
    if simplified:
        s_ood = 1 - torch.max(z_ood, dim=-1)[0]
    else:
        s_ood = ood_wass_loss(z_ood)
    tpr = sum(s_ood > threshold) / len(s_ood)
    # print(f"{tnr}: {tpr}")
    return threshold, tpr

def plot_heatmap(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D, G, method, ind_cls, ood_cls, 
                 ind_idx, ood_idx, path=None, tnr=0.99, lb=0, ub=7,m=100, f=None, simplified=False):
    # print(m)
    fig, ax = plt.subplots()
    with torch.no_grad():
        # Generated samples
        if G is not None:
            n_gen = 10
            seed = torch.rand((n_gen, 2), device=DEVICE)
            Gz = G(seed).detach().cpu().numpy()
            lb_g = np.floor(np.min(Gz)) - 1
            ub_g = np.floor(np.max(Gz)) + 1
            lb = min(lb_g, lb)
            ub = max(ub_g, ub)
        
        # m=100
        xi = np.linspace(lb, ub, m, endpoint=True)
        yi = np.linspace(lb, ub, m, endpoint=True)
        xy_pos = np.array(list(product(xi, yi)))
        zi = torch.softmax(D(torch.tensor(xy_pos, dtype=torch.float32).to(DEVICE)), dim=-1)
        # print(zi.shape)
        if simplified:
        # si = ood_wass_loss(zi)
            si = 1 - torch.max(zi, dim=-1)[0].cpu()  # Simplified wasserstein score
        else:
            si = ood_wass_loss(zi).cpu()

        threshold, _ = calculate_accuracy(D=D, ind=IND_X, ood=OOD_X, tnr=tnr)
        mask = si > threshold
    print(f"Rejection Threshold: {threshold}")
    print(f"Rejection Region Proportion: {100 * sum(mask) / len(mask):.2f}%")
    if f is not None:
        f.write(f"Rejection Threshold: {threshold}\n")
        f.write(f"Rejection Region Proportion: {100 * sum(mask) / len(mask):.2f}%\n")
    # Plot
    # Heatmap
    plt.pcolormesh(xi, yi, si.reshape((m, m)).T, shading='auto',cmap='inferno', alpha=1)
    plt.colorbar()
    plt.pcolormesh(xi, yi, mask.reshape((m, m)).T, shading='auto',cmap='gray', alpha=0.1)
    # InD and OoD
    # IND Training
    for i, idx in enumerate(ind_cls):
        if i == 0:
            plt.scatter(IND_X[:,0][IND_Y==idx][ind_idx], IND_X[:,1][IND_Y==idx][ind_idx], c='white', label ="InD", marker='^',sizes=[30]*len(IND_X), alpha=1)
        else:
            plt.scatter(IND_X[:,0][IND_Y==idx][ind_idx], IND_X[:,1][IND_Y==idx][ind_idx], c='white', marker='^',sizes=[30]*len(IND_X), alpha=1)
    # OOD BATCH
    plt.scatter(OOD_BATCH[:,0], OOD_BATCH[:,1], c='navy', label="OoD",marker='^', sizes=[30]*len(OOD_X), alpha=1)
    # IND Test
    for idx in ind_cls:
        plt.scatter(IND_X_TEST[:,0][IND_Y_TEST==idx][ind_idx], IND_X_TEST[:,1][IND_Y_TEST==idx][ind_idx], c='white', sizes=[30]*len(IND_X), alpha=0.3)
    # OOD
    for idx in ood_cls:
        plt.scatter(OOD_X[:,0][OOD_Y==idx][ood_idx], OOD_X[:,1][OOD_Y==idx][ood_idx], c='navy', sizes=[30]*len(OOD_X), alpha=0.3)

    if G is not None:
        plt.scatter(Gz[:,0], Gz[:,1], marker='x', c='#00b384', sizes=[30]*n_gen, alpha=0.5)
    plt.title(f"{method} Wasserstein Scores Heatmap")
    plt.xlabel("X1")
    plt.ylabel("X2")
    # Legend Processing
    leg = plt.legend()
    ax.add_artist(leg)
    if G is not None:
        markers = ['^', 'o', 'x']
        legends = ['Training Data', 'Testing Data', 'Generated Data']
    else:
        markers = ['^', 'o']
        legends = ['Training Data', 'Testing Data']

    h = [plt.plot([],[], color="navy", marker=mk, ls="",ms=5)[0] for mk in markers]
    plt.legend(handles=h, labels=legends, loc='lower right')
    # Save plots
    if path is None:
        plt.savefig(f"simulation_log/plot/{method}.jpg", dpi=1500)
    else:
        plt.savefig(path, dpi=1500)
    # plt.show()
    plt.close()
    # return plt

def plot_trajectory(trajectory, IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, IND_CLS, OOD_CLS, path):
    # Plot generated data
    n = int(len(IND_X) / len(IND_CLS))
    n_plot = int(n / 10) # Hardcoded, not important
    # print(n, n_plot)
    for idx in IND_CLS:
        sample_idx = np.random.choice(n, n_plot, replace=False)
        plt.scatter(IND_X[:,0][IND_Y==idx][sample_idx], IND_X[:,1][IND_Y==idx][sample_idx], label =f"InD - Class {idx+1}", sizes=[35]*len(IND_X),alpha=0.8)
    for idx in OOD_CLS:
        sample_idx = np.random.choice(n, n_plot, replace=False)
        plt.scatter(OOD_X[:,0][OOD_Y==idx][sample_idx], OOD_X[:,1][OOD_Y==idx][sample_idx], label =f"OoD - Class {idx + 1}", sizes=[35]*len(OOD_X), alpha=0.8)
    # OOD BATCH
    plt.scatter(OOD_BATCH[:,0], OOD_BATCH[:,1], c='navy', label="Observed OoD",marker='^', sizes=[30]*len(OOD_X), alpha=1)
    # Trajectory
    plt.scatter(trajectory[:,0], trajectory[:,1], label=f"G(z)", marker='x', c='#00b384', sizes=[30]*len(trajectory), alpha=0.5)
    plt.scatter(trajectory[:,0][0], trajectory[:,1][0], label=f"Start Point", marker='x', c='red', alpha=0.8)
    plt.scatter(trajectory[:,0][-1], trajectory[:,1][-1],  label=f"End Point", marker='x', c='blue', alpha=0.8)
    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Trajectory of Generated Data")
    plt.savefig(path, dpi=1500)
    plt.close()

def plot_distribution(D, IND_X, OOD_X, method):
    with torch.no_grad():
        z_ind = torch.softmax(D(torch.tensor(IND_X, dtype=torch.float32)), dim=-1)
        s_ind = ood_wass_loss(z_ind)
        z_ood = torch.softmax(D(torch.tensor(OOD_X, dtype=torch.float32)), dim=-1)
        s_ood = ood_wass_loss(z_ood)
    plt.hist(s_ind)
    plt.hist(s_ood)
    plt.legend()

def plot_loss_curve(d_loss, g_loss, path):
    ce, w_ood, w_d = d_loss[:,0], d_loss[:,1], d_loss[:,2]
    w_g, dist = g_loss[:,0], g_loss[:,1]
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

    # # Distance
    # axs[2].plot(iters, dist, label=r'Dist', marker='o', markersize=3)
    # axs[2].set_xlabel('Training Epochs')
    # axs[2].set_ylabel('Euclidean Distance')
    # axs[2].set_title("Distance Curve")
    # axs[2].legend()
    fig.savefig(path, dpi=1500)
    plt.close()

def simulate(args, config):
    # Path & Settings
    print(DEVICE)
    ckpt_dir = config['path']['ckpt_dir']
    setting = config['setting']
    # Make checkpoint directory
    dir_name = f"Eg_{args.n_ood}_[{args.h}]_[{args.beta}]_[{args.w_ce}|{args.w_ood}|{args.w_z}]_[{args.wood_lr}|{args.d_lr}|{args.g_lr}|{args.bsz_tri}|{args.bsz_val}|{args.bsz_ood}]_[{args.n_d}|{args.n_g}]"
    if args.JID is not None:
        dir_name = f"[{args.JID}]_" + dir_name

    if args.seed is not None:
        dir_name = f"[{args.seed}]_" + dir_name
    os.makedirs(os.path.join(ckpt_dir, setting, dir_name), exist_ok=True)

    f = open(os.path.join(ckpt_dir, setting, dir_name, "log.txt"), "w")
    f.write("Directory Name Format: Eg_[<n_ood>]_[<beta>]_[<CE>|<OoD>|<Z>]_[<WOOD_lr>|<D_lr>|<G_lr>|<bsz_tri>|<bsz_val>|<bsz_ood>]_[<n_d>|<n_g>]\n")
    f.write(f"Directory Name: {dir_name}\n")
    start = time.time()

    # Load InD data
    IND_CLS, OOD_CLS = torch.load(os.path.join(ckpt_dir, setting, 'data', 'ind_ood_cls.pt'))
    # X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = torch.load(os.path.join(ckpt_dir, setting, 'data', 'raw.pt'))
    IND_DATA, IND_X, IND_Y = torch.load(os.path.join(ckpt_dir, setting, 'data', 'ind_data.pt'))
    OOD_DATA, OOD_X, OOD_Y = torch.load(os.path.join(ckpt_dir, setting, 'data', 'ood_data.pt'))
    IND_DATA_TEST, IND_X_TEST, IND_Y_TEST = torch.load(os.path.join(ckpt_dir, setting, 'data', 'ind_data_test.pt'))
    # OOD_DATA_TEST, OOD_X_TEST, OOD_Y_TEST = torch.load(os.path.join(ckpt_dir, setting, 'data', 'ood_data_test.pt'))
    
    # Load OoD data
    OOD_BATCH = torch.load(os.path.join(ckpt_dir, setting, 'data', 'OoDs', f'OOD_{args.n_ood}.pt'))
    f.write(f"Observed OoD data shape: {OOD_BATCH.shape}\n")

    # WOOD SIMULATION
    wood_start = time.time()
    max_epochs = config['max_epochs']
    n_epochs_log = config['n_epochs_log']

    WOOD=True
    # WOOD=True
    if WOOD:
        f.write("\n------------- WOOD Baseline Training -------------\n") 
        f.write("WOOD Training Hyperparameters\n")
        f.write(f"h={args.h}, w_ood={args.beta}, lr={args.wood_lr}\n")
        f.write(f"epochs={max_epochs}, bsz_tri={args.bsz_tri}, bsz_val={args.bsz_val}, bsz_ood={args.bsz_ood}\n\n")
        D_WOOD = DSIM(args.h).to(DEVICE)
        optimizer = torch.optim.Adam(D_WOOD.parameters(), lr=args.wood_lr, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()
        # Dataloader
        ind_tri_loader = torch.utils.data.DataLoader(IND_DATA, shuffle=True, batch_size=args.bsz_tri)
        ind_val_loader = torch.utils.data.DataLoader(IND_DATA_TEST, shuffle=True, batch_size=args.bsz_val)
        # Training
        D_WOOD = wood_training(D_WOOD, OOD_BATCH, args.bsz_ood, args.beta, criterion, 
                            optimizer, ind_tri_loader,ind_val_loader, max_epochs, n_epochs_log, f)
        torch.save(D_WOOD.state_dict(), os.path.join(ckpt_dir, setting, dir_name, 'D_WOOD.pt'))
        # Detection Performance
        f.write("\nWOOD Performance\n")
        threshold_95, tpr_95 = calculate_accuracy(D=D_WOOD, ind=IND_X, ood=OOD_X, tnr=0.95)
        f.write(f"TPR at 95.0% TNR: {tpr_95:.4f} | Threshold at 95.0% TNR: {threshold_95}\n")
        threshold_99, tpr_99  = calculate_accuracy(D=D_WOOD, ind=IND_X, ood=OOD_X, tnr=0.99)
        f.write(f"TPR at 99.0% TNR: {tpr_99:.4f} | Threshold at 95.0% TNR: {threshold_99}\n")
        threshold_999, tpr_999  = calculate_accuracy(D=D_WOOD, ind=IND_X, ood=OOD_X, tnr=0.999)
        f.write(f"TPR at 99.9% TNR: {tpr_999:.4f} | Threshold at 95.0% TNR: {threshold_999}\n")

        # Plot
        pltargs = torch.load(os.path.join(ckpt_dir, setting, 'plt_config.pt'))
        plt_path = os.path.join(ckpt_dir, setting, dir_name, "WOOD_Heatmap.jpg")
        plot_heatmap(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D_WOOD, None, 'WOOD', 
                    IND_CLS, OOD_CLS, pltargs['ind_idx'], pltargs['ood_idx'], 
                    path=plt_path, tnr=0.95, lb=pltargs['lb'], ub=pltargs['ub'], m=pltargs['m'],f=f)
        wood_stop = time.time()
        f.write(f"WOOD Training time: {np.round(wood_stop - wood_start, 2)} s | About {np.round((wood_stop - wood_start)/60, 1)} mins\n")
    
    # OoD GAN Simulation
    gan_start = time.time()
    f.write("\n------------- Out-of-Distribution GANs Training -------------\n") 
    f.write("OoD GAN Training Hyperparameters\n")
    f.write(f"h={args.h}, w_ce={args.w_ce}, w_ood={args.w_ood}, w_z={args.w_z}, d_lr={args.d_lr}, g_lr={args.g_lr}, n_d={args.n_d}, n_g={args.n_g}\n")
    f.write(f"epochs={max_epochs}, bsz_tri={args.bsz_tri}, bsz_val={args.bsz_val}, bsz_ood={args.bsz_ood}\n\n")
    D_GAN = DSIM(args.h).to(DEVICE)
    G_GAN = GSIM(args.h).to(DEVICE)
    D_solver = torch.optim.Adam(D_GAN.parameters(), lr=args.d_lr, betas=(0.9, 0.999))
    G_solver = torch.optim.Adam(G_GAN.parameters(), lr=args.g_lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    ind_tri_loader = torch.utils.data.DataLoader(IND_DATA, shuffle=True, batch_size=args.bsz_tri)
    ind_val_loader = torch.utils.data.DataLoader(IND_DATA_TEST, shuffle=True, batch_size=args.bsz_val)
    # Training
    D_GAN, G_GAN, loss = oodgan_training(D=D_GAN, G=G_GAN, 
                                    D_solver=D_solver, 
                                    G_solver=G_solver, 
                                    OOD_BATCH=OOD_BATCH, 
                                    ood_bsz=args.bsz_ood, 
                                    bsz_tri=args.bsz_tri, 
                                    w_ce=args.w_ce, 
                                    w_wass_ood=args.w_ood,
                                    w_wass_gz=args.w_z,
                                    w_dist=None,
                                    d_step_ratio=args.n_d,
                                    g_step_ratio=args.n_g,
                                    ind_tri_loader=ind_tri_loader,
                                    ind_val_loader=ind_val_loader,
                                    max_epoch=max_epochs,
                                    n_epoch=n_epochs_log,
                                    n_step_log=25,
                                    f=f)
    
    # Plot loss relevant curve
    d_loss, g_loss, trajectory = loss
    d_loss, g_loss = np.array(d_loss), np.array(g_loss)
    loss_path = os.path.join(ckpt_dir, setting, dir_name, "OoD_GAN_Loss_Curves.jpg")
    plot_loss_curve(d_loss, g_loss, loss_path)
    
    # Save model checkpoints
    torch.save(D_GAN.state_dict(), os.path.join(ckpt_dir, setting, dir_name, 'D_GAN.pt'))
    torch.save(G_GAN.state_dict(), os.path.join(ckpt_dir, setting, dir_name, 'G_GAN.pt'))

    f.write("\nOoD GAN Performance\n")
    threshold_95, tpr_95 = calculate_accuracy(D=D_GAN, ind=IND_X, ood=OOD_X, tnr=0.95)
    f.write(f"TPR at 95.0% TNR: {tpr_95:.4f} | Threshold at 95.0% TNR: {threshold_95}\n")
    threshold_99, tpr_99  = calculate_accuracy(D=D_GAN, ind=IND_X, ood=OOD_X, tnr=0.99)
    f.write(f"TPR at 99.0% TNR: {tpr_99:.4f} | Threshold at 95.0% TNR: {threshold_99}\n")
    threshold_999, tpr_999  = calculate_accuracy(D=D_GAN, ind=IND_X, ood=OOD_X, tnr=0.999)
    f.write(f"TPR at 99.9% TNR: {tpr_999:.4f} | Threshold at 95.0% TNR: {threshold_999}\n") 

    # Plot Wasserstein Score Heatmap
    plt_path = os.path.join(ckpt_dir, setting, dir_name, "OoD_GAN_Heatmap.jpg")
    plot_heatmap(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D_GAN, G_GAN, 'OoD GAN', 
                 IND_CLS, OOD_CLS, pltargs['ind_idx'], pltargs['ood_idx'], 
                 path=plt_path, tnr=0.95, lb=pltargs['lb'], ub=pltargs['ub'], m=pltargs['m'], f=f)

    # Plot Gz trajectory plots
    plt_path = os.path.join(ckpt_dir, setting, dir_name, "OoD_GAN_G_Trajectory.jpg")
    plot_trajectory(trajectory, IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, IND_CLS, OOD_CLS, plt_path)

    gan_stop = time.time()
    f.write(f"OoD GAN Training time: {np.round(gan_stop - gan_start, 2)} s | About {np.round((gan_stop - gan_start)/60, 2)} mins | About {np.round((gan_stop - gan_start)/(60**2), 2)} hrs\n")
    
    OOD_GAN_PRETRAIN = False
    if OOD_GAN_PRETRAIN:
    # OoD GAN with WOOD pretraining
        gan_start = time.time()
        f.write("\n------------- Out-of-Distribution GANs Training (With WOOD Pretraining) -------------\n") 
        D_GAN = DSIM(args.h).to(DEVICE)
        G_GAN = GSIM(args.h).to(DEVICE)
        D_GAN.load_state_dict(D_WOOD.state_dict()) # Use pretrained weight from wood as starting point
        D_solver = torch.optim.Adam(D_GAN.parameters(), lr=args.d_lr, betas=(0.9, 0.999))
        G_solver = torch.optim.Adam(G_GAN.parameters(), lr=args.g_lr, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()
        ind_tri_loader = torch.utils.data.DataLoader(IND_DATA, shuffle=True, batch_size=args.bsz_tri)
        ind_val_loader = torch.utils.data.DataLoader(IND_DATA_TEST, shuffle=True, batch_size=args.bsz_val)
        # Training
        D_GAN, G_GAN, loss = oodgan_training(D=D_GAN, G=G_GAN, 
                                        D_solver=D_solver, 
                                        G_solver=G_solver, 
                                        OOD_BATCH=OOD_BATCH, 
                                        ood_bsz=args.bsz_ood, 
                                        bsz_tri=args.bsz_tri, 
                                        w_ce=args.w_ce, 
                                        w_wass_ood=args.w_ood,
                                        w_wass_gz=args.w_z,
                                        w_dist=None,
                                        d_step_ratio=args.n_d,
                                        g_step_ratio=args.n_g,
                                        ind_tri_loader=ind_tri_loader,
                                        ind_val_loader=ind_val_loader,
                                        max_epoch=max_epochs,
                                        n_epoch=n_epochs_log,
                                        n_step_log=25,
                                        f=f)
        
        # Plot loss relevant curve
        d_loss, g_loss, trajectory = loss
        d_loss, g_loss = np.array(d_loss), np.array(g_loss)
        loss_path = os.path.join(ckpt_dir, setting, dir_name, "OoD_GAN_Pretraining_Loss_Curves.jpg")
        plot_loss_curve(d_loss, g_loss, loss_path)
        
        # Save model checkpoints
        torch.save(D_GAN.state_dict(), os.path.join(ckpt_dir, setting, dir_name, 'D_GAN_pretrain.pt'))
        torch.save(G_GAN.state_dict(), os.path.join(ckpt_dir, setting, dir_name, 'G_GAN_pretrain.pt'))

        f.write("\nOoD GAN w/ Pretraining Performance\n")
        threshold_95, tpr_95 = calculate_accuracy(D=D_GAN, ind=IND_X, ood=OOD_X, tnr=0.95)
        f.write(f"TPR at 95.0% TNR: {tpr_95:.4f} | Threshold at 95.0% TNR: {threshold_95}\n")
        threshold_99, tpr_99  = calculate_accuracy(D=D_GAN, ind=IND_X, ood=OOD_X, tnr=0.99)
        f.write(f"TPR at 99.0% TNR: {tpr_99:.4f} | Threshold at 95.0% TNR: {threshold_99}\n")
        threshold_999, tpr_999  = calculate_accuracy(D=D_GAN, ind=IND_X, ood=OOD_X, tnr=0.999)
        f.write(f"TPR at 99.9% TNR: {tpr_999:.4f} | Threshold at 95.0% TNR: {threshold_999}\n") 

        # Plot
        plt_path = os.path.join(ckpt_dir, setting, dir_name, "OoD_GAN_Heatmap_Pretraining.jpg")
        plot_heatmap(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D_GAN, G_GAN, 'OoD GAN with Pretraining', 
                    IND_CLS, OOD_CLS, pltargs['ind_idx'], pltargs['ood_idx'], 
                    path=plt_path, tnr=0.99, lb=pltargs['lb'], ub=pltargs['ub'], m=pltargs['m'], f=f)
        
        # Plot Gz trajectory plots
        plt_path = os.path.join(ckpt_dir, setting, dir_name, "OoD_GAN_G_Trajectory_Pretraining.jpg")
        plot_trajectory(trajectory, IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, IND_CLS, OOD_CLS, plt_path)

        gan_stop = time.time()
        f.write(f"OoD GAN (w/ pretraining) Training time: {np.round(gan_stop - gan_start, 2)} s | About {np.round((gan_stop - gan_start)/60, 2)} mins | About {np.round((gan_stop - gan_start)/(60**2), 2)} hrs\n")
    
    # Stop time logging
    stop = time.time()
    f.write(f"Total time: {np.round(stop - start, 2)} s | About {np.round((stop-start)/60, 2)} mins | About {np.round((stop-start)/(60**2), 2)} hrs\n")
    f.close()

def generate_ind_ood_settings(config):
    # path
    ckpt_dir, log_dir = config['ckpt_dir'], config['log_dir']
    setting_id = config['id']
    if os.path.exists(os.path.join(ckpt_dir, setting_id, 'data')):
        assert False, f'{setting_id} already exists.'
    os.makedirs(os.path.join(ckpt_dir, setting_id, 'data'), exist_ok=True)
    # seeding
    np.random.seed(config['seed'])
    # Data config
    MU = config['data']['mu']
    COV = {}
    for k, v in config['data']['std'].items():
        COV[k] = np.eye(2)*v**2
    K = len(MU)
    SAMPLERS = get_sampler(MU, COV, K)
    # Get SAMPLES
    n = config['size']
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = get_train_test_samples(SAMPLERS=SAMPLERS, n=n)
    # Dataset
    IND_CLS, OOD_CLS = config['ind_cls'], config['ood_cls']
    IND_DATA, IND_X, IND_Y = cls_to_dset(IND_CLS, X_TRAIN, Y_TRAIN)
    OOD_DATA, OOD_X, OOD_Y = cls_to_dset(OOD_CLS, X_TRAIN, Y_TRAIN)
    IND_DATA_TEST, IND_X_TEST, IND_Y_TEST = cls_to_dset(IND_CLS, X_TEST, Y_TEST)
    OOD_DATA_TEST, OOD_X_TEST, OOD_Y_TEST = cls_to_dset(OOD_CLS, X_TEST, Y_TEST)

    # Save generated data
    torch.save((IND_CLS, OOD_CLS), os.path.join(ckpt_dir, setting_id, 'data', 'ind_ood_cls.pt'))
    torch.save((X_TRAIN, Y_TRAIN, X_TEST, Y_TEST), os.path.join(ckpt_dir, setting_id, 'data', 'raw.pt'))
    torch.save((IND_DATA, IND_X, IND_Y), os.path.join(ckpt_dir, setting_id, 'data', 'ind_data.pt'))
    torch.save((OOD_DATA, OOD_X, OOD_Y), os.path.join(ckpt_dir, setting_id, 'data', 'ood_data.pt'))
    torch.save((IND_DATA_TEST, IND_X_TEST, IND_Y_TEST), os.path.join(ckpt_dir, setting_id, 'data', 'ind_data_test.pt'))
    torch.save((OOD_DATA_TEST, OOD_X_TEST, OOD_Y_TEST), os.path.join(ckpt_dir, setting_id, 'data', 'ood_data_test.pt'))

    # Plot generated data
    n_plot = config['n_distribution']
    for idx in IND_CLS:
        sample_idx = np.random.choice(n, n_plot, replace=False)
        plt.scatter(IND_X[:,0][IND_Y==idx][sample_idx], IND_X[:,1][IND_Y==idx][sample_idx], label =f"InD - Class {idx+1}", sizes=[35]*len(IND_X),alpha=0.8)
    for idx in OOD_CLS:
        sample_idx = np.random.choice(n, n_plot, replace=False)
        plt.scatter(OOD_X[:,0][OOD_Y==idx][sample_idx], OOD_X[:,1][OOD_Y==idx][sample_idx], label =f"OoD - Class {idx + 1}", sizes=[35]*len(OOD_X), alpha=0.8)
    lb, ub = config['lb'], config['ub']
    plt.xlim((lb, ub))
    plt.ylim((lb, ub))
    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Distribution of All Simulated InD and OoD Data")
    plt.savefig(os.path.join(ckpt_dir, setting_id, 'data', 'data.jpg'), dpi=1500)

    # Generate OOD data
    os.makedirs(os.path.join(ckpt_dir, setting_id, 'data', 'OoDs'), exist_ok=True)
    for n_ood in config['ood_batch_sizes']:
        OOD_BATCH = []
        for idx in OOD_CLS:
            cls_batch = list(OOD_X[OOD_Y == idx][np.random.choice(n, n_ood, replace=False)]) 
        OOD_BATCH = OOD_BATCH + cls_batch
        OOD_BATCH = torch.tensor(np.array(OOD_BATCH), dtype=torch.float32)
        torch.save(OOD_BATCH, os.path.join(ckpt_dir, setting_id, 'data', 'OoDs', f'OOD_{n_ood}.pt'))

    # Plotting configuration generation
    n_ind, n_ood = config['n_per_ind_cls'], config['n_per_ood_cls']
    m = config['resolution']
    ind_idx = np.random.choice(n, n_ind, replace=False)
    ood_idx = np.random.choice(n, n_ood, replace=False)
    plotting_config = dict(
        ind_idx=ind_idx,
        ood_idx=ood_idx,
        lb=lb, ub=ub, m=m
    )
    torch.save(plotting_config, os.path.join(ckpt_dir, setting_id, 'plt_config.pt'))

def plot_heatmap_v2(IND_X, IND_Y, IND_X_TEST, IND_Y_TEST, OOD_X, OOD_Y, OOD_BATCH, D, G, method, ind_cls, ood_cls, 
                 ind_idx, ood_idx, title, path=None, tnr=0.99, lb=0, ub=7,m=100, f=None):
    # print(m)
    fig, ax = plt.subplots()
    with torch.no_grad():
        # Generated samples
        if G is not None:
            n_gen = 10
            seed = torch.rand((n_gen, 2), device=DEVICE)
            Gz = G(seed).detach().numpy()
            lb_g = np.floor(np.min(Gz)) - 1
            ub_g = np.floor(np.max(Gz)) + 1
            lb = min(lb_g, lb)
            ub = max(ub_g, ub)
        
        xi = np.linspace(lb, ub, m, endpoint=True)
        yi = np.linspace(lb, ub, m, endpoint=True)
        xy_pos = np.array(list(product(xi, yi)))
        zi = torch.softmax(D(torch.tensor(xy_pos, dtype=torch.float32)), dim=-1)
        # print(zi.shape)
        si = ood_wass_loss(zi)
        threshold, _ = calculate_accuracy(D=D, ind=IND_X, ood=OOD_X, tnr=tnr)
        mask = si > threshold
    print(f"Rejection Threshold: {threshold}")
    print(f"Rejection Region Proportion: {100 * sum(mask) / len(mask):.2f}%")
    if f is not None:
        f.write(f"Rejection Threshold: {threshold}\n")
        f.write(f"Rejection Region Proportion: {100 * sum(mask) / len(mask):.2f}%\n")
    # Plot
    # Heatmap
    plt.pcolormesh(xi, yi, si.reshape((m, m)).T, shading='auto',cmap='inferno', alpha=1)
    plt.colorbar()
    plt.pcolormesh(xi, yi, mask.reshape((m, m)).T, shading='auto',cmap='gray', alpha=0.15)
    # InD and OoD
    # IND Training
    for i, idx in enumerate(ind_cls):
        if i == 0:
            plt.scatter(IND_X[:,0][IND_Y==idx][ind_idx], IND_X[:,1][IND_Y==idx][ind_idx], c='white', label ="InD", marker='^',sizes=[30]*len(IND_X), alpha=1)
        else:
            plt.scatter(IND_X[:,0][IND_Y==idx][ind_idx], IND_X[:,1][IND_Y==idx][ind_idx], c='white', marker='^',sizes=[30]*len(IND_X), alpha=1)
    # OOD BATCH
    plt.scatter(OOD_BATCH[:,0], OOD_BATCH[:,1], c='navy', label="OoD",marker='^', sizes=[30]*len(OOD_X), alpha=1)
    # IND Test
    for idx in ind_cls:
        plt.scatter(IND_X_TEST[:,0][IND_Y_TEST==idx][ind_idx], IND_X_TEST[:,1][IND_Y_TEST==idx][ind_idx], c='white', sizes=[30]*len(IND_X), alpha=0.3)
    # OOD
    for idx in ood_cls:
        plt.scatter(OOD_X[:,0][OOD_Y==idx][ood_idx], OOD_X[:,1][OOD_Y==idx][ood_idx], c='navy', sizes=[30]*len(OOD_X), alpha=0.3)

    if G is not None:
        plt.scatter(Gz[:,0], Gz[:,1], marker='x', c='#00b384', sizes=[30]*n_gen, alpha=0.5)
    # plt.title(f"{method} W-Score Heatmap", fontdict={'fontsize': 14.5})
    plt.title(title, fontdict={'fontsize': 14.5})
    plt.xlabel("X1", fontdict={'fontsize': 13})
    plt.ylabel("X2", fontdict={'fontsize': 13})
    # Legend Processing
    leg = plt.legend()
    ax.add_artist(leg)
    if G is not None:
        markers = ['^', 'o', 'x']
        legends = ['Training Data', 'Testing Data', 'Generated Data']
    else:
        markers = ['^', 'o']
        legends = ['Training Data', 'Testing Data']

    h = [plt.plot([],[], color="navy", marker=mk, ls="",ms=5)[0] for mk in markers]
    plt.legend(handles=h, labels=legends, loc='lower right')
    # Save plots
    if path is None:
        plt.savefig(f"simulation_log/plot/{method}.jpg", dpi=1500)
    else:
        plt.savefig(path, dpi=1500)
    # plt.show()
    plt.close()

# Example command for G mode:
# python3 simulation.py --mode=G --config=config/simulation/setting_1_config.yaml

# Example command for R mode:
# python3 simulation.py --config=config/simulation/R_config.yaml --mode=R --JID=0 
# --n_ood=32 --h=128 --beta=1 --w_ce=1 --w_ood=1 --w_z=1 --wood_lr=0.001 --d_lr=0.0001 
# --g_lr=0.0001 --bsz_tri=256 --bsz_val=256 --bsz_ood=4 --n_d=1 --n_g=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='G - Generation | R - Run')
    parser.add_argument('--config', help='Configuration file')
    # NN structures
    parser.add_argument('--h', help="Number of NN hidden dimensions", type=int)
    # Weight
    parser.add_argument('--beta', help="WOOD beta", type=float)
    parser.add_argument('--w_ce', help='CE term weight', type=float)
    parser.add_argument('--w_ood', help="OoD term weight", type=float)
    parser.add_argument('--w_z', help="Adversarial term weight", type=float)
    # Training parameters
    parser.add_argument('--seed', help="seed", type=int, default=1)
    parser.add_argument('--n_ood', help="Number of OoD samples", type=int)
    parser.add_argument('--wood_lr', help="WOOD learning rate", type=float)
    parser.add_argument('--d_lr', help="OoD GAN Discriminator learning rate", type=float)
    parser.add_argument('--g_lr', help="OoD GAN Generator learning rate", type=float)
    parser.add_argument('--bsz_tri', help="Training batch size", type=int)
    parser.add_argument('--bsz_val', help="Validation batch size", type=int)
    parser.add_argument('--bsz_ood', help='OoD batch size', type=int)
    parser.add_argument('--n_d', help='d_step_ratio', type=int)
    parser.add_argument('--n_g', help='g_step_ratio', type=int)

    # Productivity
    parser.add_argument('--JID', help='GL Job ID', type=str)
    
    args = parser.parse_args()
    assert args.config is not None, 'Please specify the config .yml file to proceed.'
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #     print(f"SEED: {args.seed}")
    # else:
    #     print("Setting seed for reproducibility.")
    #     torch.manual_seed(1)
    #     np.random.seed(1)

    if args.mode == "G":
        generate_ind_ood_settings(config)
    elif args.mode == 'R':
        setting = config['setting']
        ckpt_dir = config['path']['ckpt_dir']
        assert os.path.exists(os.path.join(ckpt_dir, setting))
        simulate(args, config)
    else:
        assert False, 'Unrecognized mode.'


