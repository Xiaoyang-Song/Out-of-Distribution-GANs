from config import *
# Auxiliary imports
from utils import visualize_img


def MNIST(batch_size, test_batch_size, num_workers=0, shuffle=True):

    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle,
                                               batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_set, shuffle=shuffle,
                                              batch_size=test_batch_size,  num_workers=num_workers)

    return train_set, val_set, train_loader, test_loader


def MNIST_SUB(batch_size: int, val_batch_size: int, idx_ind: list, idx_ood: list, shuffle=True):
    """
    Helper function to extract subset of the MNIST dataset. In short, return training 
    and validation sets with labels specified in 'idx_ind' and 'idx_ood', respectively. For 
    samples with other labels, just ignore them.

    Args:
        batch_size (int): training batch size.
        val_batch_size (int): validation batch size.
        idx_ind (list): a list of integer from 0-9 (specifying in-distribution labels)
        idx_ood (list): a list of integer from 0-9 (specifying out-of-distribution labels)
        shuffle (bool, optional): whether or not to shuffle the dataset. Defaults to True.
    """
    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    return train_set, val_set
    # pass


def CIFAR10(batch_size, test_batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Resize(size=28)])
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
    # train_dataset, val_dataset, train_loader_mnist, val_loader_mnist = MNIST(
    #     32, 16)
    # # Test single sample:
    # ic(len(train_dataset.__getitem__(0)))  # First sample in the loader
    # ic(train_dataset.__getitem__(0)[0].shape)  # img features
    # ic(train_dataset.__getitem__(0)[1])  # Class label
    # mnist_sample = train_dataset.__getitem__(0)[0]
    # # Show the sample image
    # # plt.imshow(mnist_sample.squeeze(), cmap="gray")
    # # plt.show()
    # # ic(mnist_sample)

    # VISUALIZE CIFAR-10 dataset
    # train_dataset, val_dataset, train_loader_mnist, val_loader_mnist = CIFAR10(
    #     32, 16)
    # ic(train_dataset.__getitem__(0)[0].shape)  # img features
    # cifar_sample_grayscale = train_dataset.__getitem__(7)[0].mean(0)
    # cifar_sample_label = train_dataset.__getitem__(7)[1]
    # ic(cifar_sample_label)  # By manually inspection, this should be a horse
    # plt.imshow(cifar_sample_grayscale.squeeze(), cmap="gray")
    # plt.show()

    # MNIST_SUB Sanity Check
    train_set, val_set = MNIST_SUB(
        128, 64, idx_ind=[0, 2, 3, 6, 8], idx_ood=[1, 7])
    ic(train_set.targets.shape)
    ic(train_set.data.shape)
    ic(type(train_set.data))
    train_loader = torch.utils.data.DataLoader(
        train_set.data, batch_size=128, shuffle=True)
    ic(train_loader)
