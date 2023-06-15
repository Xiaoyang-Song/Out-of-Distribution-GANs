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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, z):
        return self.fc2(self.relu(self.fc1(z)))
        

class DSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        # self.fc = nn.Linear(2, 3)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        # return self.fc(x)
    

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
            labels = labels.to(DEVICE) - 1
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
                img, labels = img.to(torch.float32), labels.to(DEVICE) - 1
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
            labels = labels.to(DEVICE) - 1
            optimizer.zero_grad()
            logits = D(img)
            # print(labels)
            ood_idx = np.random.choice(
            len(ood_batch), min(len(ood_batch), ood_bsz), replace=False)
            ood_samples = ood_batch[ood_idx, :].to(DEVICE)
            ood_logits = D(ood_samples)
            # ic(ood_logits.shape)
            wass_loss = batch_wasserstein(ood_logits)
            loss = criterion(logits, labels) + beta * wass_loss
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
                img, labels = img.to(torch.float32), labels.to(DEVICE) - 1
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

def oodgan_training(D, G, D_solver, G_solver, OOD_BATCH, ood_bsz, bsz_tri, w_ce, w_wass, \
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
    
    def ood_gan_g_loss(logits_fake):
        # 1. Wasserstein distance of G(z)
        w_fake = batch_wasserstein(logits_fake)
        return w_fake
    
    iter_count = 0
    ood_batch = OOD_BATCH.to(DEVICE)
    for epoch in range(max_epoch):
        D.train()
        for steps, (x, y) in enumerate(tqdm(ind_tri_loader)):
            x,y = x.to(torch.float32), y.to(DEVICE) - 1
            # ---------------------- #
            # DISCRIMINATOR TRAINING #
            # ---------------------- #
            D_solver.zero_grad()
            # Logits for X_in
            logits_real = D(x)

            seed = torch.rand((bsz_tri, 1), device=DEVICE) * 2 - 1
            # Gz = self.G(seed, [cls]*self.bsz_tri).detach()

            Gz = G(seed).detach()

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
            d_total = w_ce * ind_ce_loss + \
                w_wass * (w_ood - (w_fake))
            # Update
            d_total.backward()
            D_solver.step()

            # ------------------ #
            # GENERATOR TRAINING #
            # ------------------ #
            G_solver.zero_grad()
            logits_fake = D(Gz)
            w_z = ood_gan_g_loss(logits_fake)
            g_total = -w_wass * (w_z)
            # Update
            g_total.backward()
            G_solver.step()

            # Print out statistics
            if (iter_count % n_step_log == 0):
                print(
                    f"Step: {steps:<4} | D: {d_total.item(): .4f} | CE: {ind_ce_loss.item(): .4f} | W_OoD: {-torch.log(-w_ood).item(): .4f} | W_z: {-torch.log(-w_fake).item(): .4f} | G: {g_total.item(): .4f} | W_z: {-torch.log(-w_z).item(): .4f}")
            iter_count += 1

        D.eval()
        with torch.no_grad():
            val_acc = []
            for idx, (img, labels) in enumerate(ind_val_loader):
                img, labels = img.to(torch.float32), labels.to(DEVICE) - 1
                logits = D(img)
                acc = (torch.argmax(logits, dim=1) ==
                    labels).sum().item() / labels.shape[0]
                val_acc.append(acc)
            if epoch % n_epoch == 0:
                print(f"Epoch  # {epoch + 1} | Val accuracy: {np.round(np.mean(val_acc), 4)}")

def plot_heatmap(ind, ood, observed_ood, D):
    plt.scatter(ind[:,0], ind[:,1], c='orange', label ="InD", alpha=1)
    plt.scatter(ood[:,0], ood[:,1], c='navy', label="OoD", alpha=0.05)
    plt.scatter(observed_ood[:,0], observed_ood[:,1], c='navy', label="OoD", alpha=1)
    xi = np.linspace(0, 8, 50, endpoint=True)
    yi = np.linspace(0, 8, 50, endpoint=True)
    xy_pos = np.array(list(product(xi, yi)))
    zi = torch.softmax(D(torch.tensor(xy_pos, dtype=torch.float32)), dim=-1)
    # print(zi.shape)
    si = ood_wass_loss(zi)
    plt.pcolormesh(xi, yi, si.reshape((50,50)).T, shading='auto', alpha=0.8)
    plt.title("Plot Title")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()