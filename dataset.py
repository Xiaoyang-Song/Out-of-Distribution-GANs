from config import *

#


def MNIST(batch_size, test_batch_size):

    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size)

    return train_loader, test_loader


def Cifar_10(batch_size, test_batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    # std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data/cifar10',
                                                                train=True, download=True,
                                                                transform=transform),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./datasets/cifar10', train=False, download=True,
                                                              transform=transform),
                                             batch_size=test_batch_size, shuffle=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, test_loader = MNIST(32, 16)
