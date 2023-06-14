import torch
import numpy as np
from numpy.random import multivariate_normal as mn
import torch.nn as nn
from tqdm import tqdm
from config import DEVICE



class N():
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
    def sample(self, n):
        return mn(self.mu, self.cov, n)

class GSIM(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 2)
    
    def forward(self, z):
        return self.fc2(self.relu(self.fc1))
        

class DSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,3)
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