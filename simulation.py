import torch
import numpy as np
from numpy.random import multivariate_normal as mn
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from config import DEVICE
from itertools import product
from wasserstein import batch_wasserstein, ood_wass_loss
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

def wood_training(D, OOD_BATCH, ood_bsz, beta, criterion, optimizer, ind_tri_loader, ind_val_loader, max_epoch, n_epoch=10):
    # Simple training loop
    ood_batch = OOD_BATCH.to(DEVICE)
    iter_count_train = 0
    iter_count_val = 0
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
            wass_loss = batch_wasserstein(ood_logits)
            # print(wass_loss)
            # loss = wass_loss
            # print(loss)
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
            print(wass_loss)
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
    return D

def oodgan_training(D, G, D_solver, G_solver, OOD_BATCH, ood_bsz, bsz_tri, w_ce, w_wass_ood, w_wass_gz, w_dist, \
                    d_step_ratio, g_step_ratio, ind_tri_loader, ind_val_loader, max_epoch, n_epoch=10, n_step_log = 100):
    
    assert d_step_ratio == 1 or g_step_ratio == 1

    def ood_gan_d_loss(logits_real, logits_fake, logits_ood, labels_real):
        # 1: CrossEntropy of X_in
        criterion = nn.CrossEntropyLoss()
        ind_ce_loss = criterion(logits_real, labels_real)
        # 2. W_ood
        assert logits_ood.requires_grad
        w_ood = batch_wasserstein(logits_ood)
        # 3. W_z
        assert logits_fake.requires_grad
        w_fake = batch_wasserstein(logits_fake)
        return ind_ce_loss, w_ood, w_fake
    
    def ood_gan_g_loss(logits_fake, gz, xood):
        # 1. Wasserstein distance of G(z)
        assert logits_fake.requires_grad
        w_fake = batch_wasserstein(logits_fake)
        # distance term
        # print(torch.mean(gz, dim=0) - torch.mean(xood, dim=0))
        dist = torch.sqrt(torch.sum((torch.mean(gz) - torch.mean(xood))**2))
        assert dist.requires_grad
        return w_fake, dist
    
    iter_count = 0
    ood_batch = OOD_BATCH.to(DEVICE)
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
            iter_count += 1

        D.eval()
        with torch.no_grad():
            val_acc = []
            for idx, (img, labels) in enumerate(ind_val_loader):
                img, labels = img.to(torch.float32), labels.to(DEVICE)
                logits = D(img)
                acc = (torch.argmax(logits, dim=1) ==
                    labels).sum().item() / labels.shape[0]
                val_acc.append(acc)
            if epoch % n_epoch == 0:
                print(f"Epoch  # {epoch + 1} | Val accuracy: {np.round(np.mean(val_acc), 4)}")

def calculate_accuracy(D, ind, ood, tnr):
    z = torch.softmax(D(torch.tensor(ind, dtype=torch.float32)), dim=-1)
    # print(z.shape)
    s = ood_wass_loss(z)
    # print(s.shape)
    threshold = np.quantile(s, tnr)
    # print(threshold)
    z_ood = torch.softmax(D(torch.tensor(ood, dtype=torch.float32)), dim=-1)
    # print(z_ood.shape)
    s_ood = ood_wass_loss(z_ood)
    tpr = sum(s_ood > threshold) / len(s_ood)
    print(f"{tnr}: {tpr}")
    return threshold

def plot_heatmap(IND_X, IND_X_TEST, OOD_X, OOD_BATCH, D, G, method, ind_idx, ood_idx, m=100):
    # print(m)
    fig, ax = plt.subplots()
    with torch.no_grad():
        xi = np.linspace(0, 7, m, endpoint=True)
        yi = np.linspace(0, 7, m, endpoint=True)
        xy_pos = np.array(list(product(xi, yi)))
        zi = torch.softmax(D(torch.tensor(xy_pos, dtype=torch.float32)), dim=-1)
        # print(zi.shape)
        si = ood_wass_loss(zi)
        threshold = calculate_accuracy(D=D, ind=IND_X, ood=OOD_X, tnr=0.99)
        mask = si > threshold
    print(f"Rejection Threshold: {threshold}")
    print(f"Rejection Region Proportion: {100 * sum(mask) / len(mask):.2f}%")   
    # Plot
    # Heatmap
    plt.pcolormesh(xi, yi, si.reshape((m, m)).T, shading='auto',cmap='inferno', alpha=1)
    plt.colorbar()
    plt.pcolormesh(xi, yi, mask.reshape((m, m)).T, shading='auto',cmap='gray', alpha=0.1)
    # InD and OoD
    plt.scatter(IND_X[:,0][ind_idx], IND_X[:,1][ind_idx], c='white', label ="InD", marker='^',sizes=[30]*len(IND_X), alpha=1)
    plt.scatter(OOD_BATCH[:,0], OOD_BATCH[:,1], c='navy', label="OoD",marker='^', sizes=[30]*len(OOD_X), alpha=1)
    plt.scatter(IND_X_TEST[:,0][ind_idx], IND_X_TEST[:,1][ind_idx], c='white', sizes=[30]*len(IND_X), alpha=0.3)
    plt.scatter(OOD_X[:,0][ood_idx], OOD_X[:,1][ood_idx], c='navy', sizes=[30]*len(OOD_X), alpha=0.3)

    # Generated samples
    if G is not None:
        n_gen = 10
        seed = torch.rand((n_gen, 2), device=DEVICE)
        Gz = G(seed).detach().numpy()
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
    plt.legend(handles=h, labels=legends, loc='lower left')
    # Save plots
    plt.savefig(f"simulation_log/plot/{method}.jpg", dpi=1500)
    # plt.show()
    # return plt

def plot_distribution(D, IND_X, OOD_X, method):
    with torch.no_grad():
        z_ind = torch.softmax(D(torch.tensor(IND_X, dtype=torch.float32)), dim=-1)
        s_ind = ood_wass_loss(z_ind)
        z_ood = torch.softmax(D(torch.tensor(OOD_X, dtype=torch.float32)), dim=-1)
        s_ood = ood_wass_loss(z_ood)
    plt.hist(s_ind)
    plt.hist(s_ood)
    plt.legend()

def simulate(config):
    pass

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
    ind_idx = np.random.choice(len(IND_X), n_ind, replace=False)
    ood_idx = np.random.choice(len(OOD_X), n_ood, replace=False)
    plotting_config = dict(
        ind_idx=ind_idx,
        ood_idx=ood_idx,
        lb=lb, ub=ub, m=m
    )
    torch.save(plotting_config, os.path.join(ckpt_dir, setting_id, 'plt_config.pt'))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='G - Generation | R - Run')
    parser.add_argument('--config', help='Configuration file')
    # NN structures
    parser.add_argument('--h', help="Number of NN hidden dimensions", type=int)
    # Weight
    parser.add_argument('--w_ce', help='CE term weight', type=float)
    parser.add_argument('--w_ood', help="OoD term weight", type=float)
    parser.add_argument('--w_z', help="Adversarial term weight", type=float)
    # Training parameters
    parser.add_argument('--lr', help="Learning rate", type=float)
    parser.add_argument('--bsz_tri', help="Training batch size", type=float)
    parser.add_argument('--bsz_val', help="Validation batch size", type=float)
    parser.add_argument('--bsz_ood', help='OoD batch size', type=int)
    parser.add_argument('--n_d', help='d_step_ratio', type=int)
    parser.add_argument('--n_g', help='g_step_ratio', type=int)
    
    args = parser.parse_args()
    assert args.config is not None, 'Please specify the config .yml file to proceed.'
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    if args.mode == "G":
        generate_ind_ood_settings(config)
    elif args.mode == 'R':
        setting = config['id']
        ckpt_dir = config['path']['ckpt_dir']
        assert os.path.exists(os.path.join(ckpt_dir, setting))
        simulate(config)
    else:
        assert False, 'Unrecognized mode.'


