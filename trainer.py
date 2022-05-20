from config import *
from models.mnist_cnn import MNISTCNN


def train(model, train_loader, val_loader, num_epoch=8):
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
            optimizer.zero_grad()
            img, label = img.to(DEVICE), label.to(DEVICE)
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            # Append training statistics
            train_acc.append(
                (torch.argmax(logits, dim=1) == label).sum().item())
            train_loss.append(loss.detach().item())

        print(f"Epoch  # {epoch + 1} | training loss: {torch.mean(train_loss)} \
              | training acc: {torch.mean(train_acc)}")
        # Evaluation
        model.eval()
        with torch.no_grad:
            # TODO: print more stats for every n epoch: fix this later.
            val_loss, val_acc = [], []
            for idx, (img, label) in enumerate(val_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                logits = model(img)
                loss = criterion(logits, label)
                val_acc.append(
                    (torch.argmax(logits, dim=1) == label).sum().item())
                val_loss.append(loss.detach().item())
            print(f"Epoch  # {epoch + 1} | training loss: {torch.mean(val_loss)} \
              | training acc: {torch.mean(val_acc)}")


def get_optimizer(model):
    # TODO: Make it more generic later.
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))


def get_criterion():
    # TODO: Make it more generic later.
    return torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    pass
