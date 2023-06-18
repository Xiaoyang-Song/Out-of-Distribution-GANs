import torch
import numpy as np
from numpy.random import multivariate_normal as mn
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from config import DEVICE
from itertools import product
from wasserstein import batch_wasserstein, ood_wass_loss



class N():
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
    def sample(self, n):
        return mn(self.mu, self.cov, n)

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
            img = img.to(torch.float32)
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

def oodgan_training(D, G, D_solver, G_solver, OOD_BATCH, ood_bsz, bsz_tri, w_ce, w_wass, w_dist,\
                    ind_tri_loader, ind_val_loader, max_epoch, n_epoch=10, n_step_log = 100):
    
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
            x,y = x.to(torch.float32), y.to(DEVICE)
            # ---------------------- #
            # DISCRIMINATOR TRAINING #
            # ---------------------- #
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
            d_total = w_ce * ind_ce_loss - w_wass * (w_ood - 0.1 * w_fake)
            # Update
            d_total.backward()
            D_solver.step()

            # ------------------ #
            # GENERATOR TRAINING #
            # ------------------ #
            for g_step in range(5):
                seed = torch.rand((bsz_tri, 2), device=DEVICE)
                # Gz = self.G(seed, [cls]*self.bsz_tri).detach()

                Gz = G(seed)
                G_solver.zero_grad()
                logits_fake = D(Gz)
                w_z, dist = ood_gan_g_loss(logits_fake, Gz, ood_sample)
                # g_total = -w_wass * (w_z) + dist * w_dist
                g_total = -w_wass * 0.1 * (w_z)
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

def plot_heatmap(IND_X, IND_X_TEST, OOD_X, OOD_BATCH, D, method, m=100, n_ind=25, n_ood=25):
    xi = np.linspace(0, 6, m, endpoint=True)
    yi = np.linspace(0, 6, m, endpoint=True)
    xy_pos = np.array(list(product(xi, yi)))
    zi = torch.softmax(D(torch.tensor(xy_pos, dtype=torch.float32)), dim=-1)
    print(zi.shape)
    si = ood_wass_loss(zi)
    plt.pcolormesh(xi, yi, si.reshape((m, m)).T, shading='auto', alpha=0.8)
    plt.colorbar()
    # InD and OoD
    ind_idx = np.random.choice(len(IND_X), n_ind, replace=False)
    ood_idx = np.random.choice(len(OOD_X), n_ind, replace=False)
    plt.scatter(IND_X[:,0][ind_idx], IND_X[:,1][ind_idx], c='orange', label ="InD", sizes=[3]*len(IND_X), alpha=1)
    plt.scatter(OOD_BATCH[:,0], OOD_BATCH[:,1], c='black', label="OoD", sizes=[3]*len(OOD_X), alpha=1)
    plt.scatter(IND_X_TEST[:,0][ind_idx], IND_X_TEST[:,1][ind_idx], c='orange', sizes=[3]*len(IND_X), alpha=0.2)
    plt.scatter(OOD_X[:,0][ood_idx], OOD_X[:,1][ood_idx], c='black', sizes=[2]*len(OOD_X), alpha=0.05)
    plt.title(f"{method} InD/OoD Separation Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"simulation_log/plot/{method}.jpg", dpi=1000)
    # plt.show()
    return plt

def plot_scatter(ind):
    pass



