from config import *
# Auxiliary imports
from utils import visualize_img


def MNIST(batch_size, test_batch_size):

    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(val_set,
                                              batch_size=test_batch_size)

    return train_set, val_set, train_loader, test_loader


def CIFAR10(batch_size, test_batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    # std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    train_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':
    # Test dataset functions
    train_dataset, val_dataset, train_loader_mnist, val_loader_mnist = MNIST(
        32, 16)
    # Test single sample:
    ic(len(train_dataset.__getitem__(0)))  # First sample in the loader
    ic(train_dataset.__getitem__(0)[0].shape)  # img features
    ic(train_dataset.__getitem__(0)[1])  # Class label
    mnist_sample = train_dataset.__getitem__(0)[0]
    # Show the sample image
    plt.imshow(mnist_sample.squeeze(), cmap="gray")
    plt.show()
    # ic(mnist_sample)
