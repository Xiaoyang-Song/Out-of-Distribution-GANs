from models.dc_gan_model import *
from dataset import *
from config import *
from models.model import *
from eval import *
import argparse
import time
import yaml

log_dir = "checkpoint/pretrained/"
os.makedirs(log_dir, exist_ok=True)
# model = DenseNet3(depth=100, num_classes=8, input_channel=3).to(DEVICE)
img_info = {'H': 28, 'W': 28, 'C': 1}
model = DC_D(8, img_info).to(DEVICE)

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
# CIFAR10-SVHN
# dset = DSET("CIFAR10-SVHN", False, 50, 64, None, None)
# MNIST
# dset = DSET("MNIST", True, 50, 256, [2, 3, 6, 8, 9], [1, 7])
# SVHN
# dset = DSET("SVHN", True, 50, 64, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
dset = DSET("FashionMNIST", True, 50, 64, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])


ind_tri_loader = dset.ind_train_loader
ind_val_loader = dset.ind_val_loader
max_epoch = 50


criterion = nn.CrossEntropyLoss()
# Simple training loop
iter_count_train = 0
iter_count_val = 0
for epoch in tqdm(range(max_epoch)):
    # Training
    model.train()
    train_loss, train_acc, wass = [], [], []
    for idx, (img, labels) in enumerate(ind_tri_loader):
        img = img.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(img)
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

    # pretrain_writer.add_scalar("Training/Accuracy (Epoch)", np.mean(train_acc), epoch)
    # pretrain_writer.add_scalar("Training/Loss (Epoch)", np.mean(train_loss), epoch)
    print(f"\nEpoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
            | training acc: {np.mean(train_acc)}")
    # Evaluation
    # scheduler.step()
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = [], []
        for idx, (img, labels) in enumerate(ind_val_loader):
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            logits = model(img)
            loss = criterion(logits, labels)
            acc = (torch.argmax(logits, dim=1) ==
                   labels).sum().item() / labels.shape[0]
            val_acc.append(acc)
            val_loss.append(loss.detach().item())
            iter_count_val += 1

        print(f"Epoch  # {epoch + 1} | validation loss: {np.mean(val_loss)} \
            | validation acc: {np.mean(val_acc)}")
with torch.no_grad():
    # Evaluation
    torch.save(model.state_dict(),
               log_dir + f"[{dset.name}]-pretrained-classifier.pt")
    ic("Model Checkpoint Saved!")
    test_backbone_D(model, dset.ind_val_loader)
