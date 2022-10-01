from config import *
from models.mnist_cnn import MNISTCNN
from tqdm import tqdm


def train(model, train_loader, val_loader=None, num_epoch=8, device=DEVICE):
    # TODO: Make it more genetic later.
    # model = MNISTCNN().to(DEVICE)
    optimizer = get_optimizer(model)
    criterion = get_criterion()
    # Simple training loop
    for epoch in tqdm(range(num_epoch)):
        # Training
        model.train()
        train_loss, train_acc = [], []
        for idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            # Append training statistics
            train_acc.append(
                (torch.argmax(logits, dim=1) == label).sum().item() / label.shape[0])
            train_loss.append(loss.detach().item())

        print(f"Epoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
              | training acc: {np.mean(train_acc)}")
        # Evaluation
        model.eval()
        with torch.no_grad():
            if val_loader is None:
                print(f"No validation now.")
                break
            # TODO: print more stats for every n epoch: fix this later.
            val_loss, val_acc = [], []
            for idx, (img, label) in enumerate(val_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                logits = model(img)
                loss = criterion(logits, label)
                val_acc.append(
                    (torch.argmax(logits, dim=1) == label).sum().item() / label.shape[0])
                val_loss.append(loss.detach().item())
            print(f"Epoch  # {epoch + 1} | validation loss: {np.mean(val_loss)} \
              | validation acc: {np.mean(val_acc)}")
    return model


def get_optimizer(model):
    # TODO: Make it more generic later.
    lr = 1e-3
    ic(f"Learning rate = {lr}")
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


def get_criterion():
    # TODO: Make it more generic later.
    return torch.nn.CrossEntropyLoss()
    # return torch.nn.BCELoss()


if __name__ == "__main__":
    ic("Hello trainer.py")
