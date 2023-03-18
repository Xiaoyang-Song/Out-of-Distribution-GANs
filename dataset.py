from random import sample
from config import *
# Auxiliary imports
from utils import visualize_img
from collections import defaultdict, Counter


def FashionMNIST(bs_t, bs_v, sf):
    tset = torchvision.datasets.FashionMNIST(
        "./Datasets", download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
    vset = torchvision.datasets.FashionMNIST(
        "./Datasets", download=False, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    # Get data loader
    t_loader = torch.utils.data.DataLoader(tset, shuffle=sf, batch_size=bs_t)
    v_loader = torch.utils.data.DataLoader(vset, shuffle=sf, batch_size=bs_v)
    return tset, vset, t_loader, v_loader


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
    def get_subsamples(label_idx: list[list, list], dset):
        assert len(label_idx) == 2, 'Expect a nested list for label_idx'
        assert len(label_idx[0]) + len(label_idx[1]
                                       ) <= 10, 'Two lists should be less than length of 10 in total'
        # TODO: Change this to make it more generic later.
        ind_sub_idx, ood_sub_idx = [torch.tensor(list(filterfalse(
            lambda x: dset.targets[x] not in idx, torch.arange(dset.data.shape[0])))) for idx in label_idx]
        # ind_sub, ood_sub = [[(dset.data[idx], dset.targets[idx])]
        #                     for idx in [ind_sub_idx, ood_sub_idx]]
        ind_sub, ood_sub = [[dset.__getitem__(idx) for idx in idxs]
                            for idxs in [ind_sub_idx, ood_sub_idx]]
        return ind_sub, ood_sub

    train_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_set = torchvision.datasets.MNIST(
        "./Datasets", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

    train_sub, val_sub = [get_subsamples([idx_ind, idx_ood], dset) for dset in [
        train_set, val_set]]
    train_set_ind, train_set_ood = train_sub
    val_set_ind, val_set_ood = val_sub
    # Build pytorch dataloaders

    def set_to_loader(dset: torch.tensor, bs: int, sf: bool):
        return torch.utils.data.DataLoader(dset, batch_size=bs, shuffle=sf)
    dset_dict = {
        'train_set_ind': train_set_ind,
        'train_set_ood': train_set_ood,
        'val_set_ind': val_set_ind,
        'val_set_ood': val_set_ood,
        'train_set_ind_loader': set_to_loader(train_set_ind, batch_size, shuffle),
        'train_set_ood_loader': set_to_loader(train_set_ood, batch_size, shuffle),
        'val_set_ind_loader': set_to_loader(val_set_ind, val_batch_size, shuffle),
        'val_set_ind_loader': set_to_loader(val_set_ind, val_batch_size, shuffle)
    }
    return dset_dict


def CIFAR10(batch_size, test_batch_size):

    # Ground truth mean & std:
    # mean = torch.tensor([125.3072, 122.9505, 113.8654])
    # std = torch.tensor([62.9932, 62.0887, 66.7049])
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])
    train_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=True,
                                     download=True, transform=transform)
    # ic(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader


def SVHN(bsz_tri, bsz_val, shuffle=True):

    # Ground truth mean & std
    # mean = torch.tensor([111.6095, 113.1610, 120.5650])
    # std = torch.tensor([50.4977, 51.2590, 50.2442])
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [111.6095, 113.1610, 120.5650]],
                                      std=[x/255.0 for x in [50.4977, 51.2590, 50.2442]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])

    # Load dataset & Loader
    train_dataset = datasets.SVHN('./Datasets/SVHN', split='train',
                                  download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz_tri, shuffle=shuffle)
    val_dataset = datasets.SVHN('./Datasets/SVHN', split='test', download=True,
                                transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz_val, shuffle=shuffle)

    return train_dataset, val_dataset, train_loader, val_loader


def MNIST_By_CLASS(train):
    mnist_tri = torchvision.datasets.MNIST(
        "./Datasets", train=train, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    ic(len(mnist_tri))
    img_lst = defaultdict(list)
    label_lst = defaultdict(list)
    # Loop through each tuple
    for item in mnist_tri:
        img_lst[item[1]].append(item[0])
        label_lst[item[1]].append(item[1])
    # Declare a wrapper dictionary
    mnist = {}
    for label in np.arange(10):
        mnist[label] = (img_lst[label], label_lst[label])
    return mnist

# Specifically for Ind Ood Separation


def form_ind_dsets(input_dsets, ind_idx):
    dset = []
    for label in ind_idx:
        dset += list(zip(input_dsets[label][0], input_dsets[label][1]))
    return dset


def sample_from_ood_class(mnist: dict, ood_idx: list, sample_size):
    samples = []
    for idx in ood_idx:
        img, label = mnist[idx]
        rand_idx = np.random.choice(len(label), sample_size, False)
        x, y = [img[i] for i in rand_idx], [label[i] for i in rand_idx]
        samples += list(zip(x, y))
    return samples


def set_to_loader(dset: torch.tensor, bs: int, sf: bool):
    return torch.utils.data.DataLoader(dset, batch_size=bs, shuffle=sf)


def relabel_tuples(dsets, ori, target):
    transformation = dict(zip(ori, target))
    transformed = []
    for dpts in dsets:
        transformed.append((dpts[0], transformation[dpts[1]]))
    return transformed


def check_classes(dset):
    ic(Counter(list(zip(*dset))[1]))


def tuple_list_to_tensor(dset):
    x = torch.stack([data[0] for data in dset])
    y = torch.tensor([data[1] for data in dset])
    return x, y


class DSET():
    def __init__(self, name, bsz_tri, bsz_val, ind=None, ood=None):
        self.bsz_tri = bsz_tri
        self.bsz_val = bsz_val
        self.ind, self.ood = ind, ood
        if name == 'mnist':
            assert ind is not None and ood is not None
            self.train = MNIST_By_CLASS(train=True)
            self.val = MNIST_By_CLASS(train=False)
            self.ind_train = form_ind_dsets(self.train, ind)
            self.ind_val = form_ind_dsets(self.val, ind)
            self.ood_train = form_ind_dsets(self.train, ood)
            self.ood_val = form_ind_dsets(self.val, ood)
            self.ind_train = relabel_tuples(
                self.ind_train, ind, np.arange(len(ind)))
            self.ind_val = relabel_tuples(
                self.ind_val, ind, np.arange(len(ind)))
            self.ind_train_loader = set_to_loader(
                self.ind_train, self.bsz_tri, True)
            self.ind_val_loader = set_to_loader(
                self.ind_val, self.bsz_val, True)

    def get_ood_equal(self, n):
        ood_sample = sample_from_ood_class(self.train, self.ood, n)
        ood_img_batch, ood_img_label = tuple_list_to_tensor(ood_sample)
        return ood_img_batch, ood_img_label

    def get_ood_unequal(self, idx, n):  # Note that this function is for MNIST only
        ood_sample = sample_from_ood_class(self.train, [self.ood[idx]], n)
        ood_img_batch, ood_img_label = tuple_list_to_tensor(ood_sample)
        return ood_img_batch, ood_img_label


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
    # train_set, val_set = MNIST_SUB(
    #     128, 64, idx_ind=[0, 2, 3, 6, 8], idx_ood=[1, 7])
    # ic(train_set.targets.shape)
    # mask = train_set.targets == 0
    # ic((train_set.targets[mask]).shape)
    # ic(torch.tensor(list(filterfalse(
    #     lambda x: train_set.targets[x] in [0, 1], torch.arange(60000)))).shape)
    # ic(train_set.data.shape)
    # ic(type(train_set.data))
    # train_loader = torch.utils.data.DataLoader(
    #     train_set.data, batch_size=128, shuffle=True)
    # ic(train_loader)

    # Test whether the data loader is randomly shuffled in general.
    # idx_ind = [0, 1, 3, 4, 5]
    # dset_dict = MNIST_SUB(batch_size=128, val_batch_size=64,
    #                       idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    # ic(len(dset_dict['train_set_ind'][0]))
    # ic(dset_dict['train_set_ind'][0][0].shape)
    # # ic(dset_dict['train_set_ind'][0][1].shape)
    # ic(next(iter(dset_dict['train_set_ind_loader']))[1])
    # batch = next(iter(dset_dict['train_set_ind_loader']))[1]
    # for num in idx_ind:
    #     ic(len(batch[batch == num]))

    # Test FashionMNIST
    # tset, vset, t_loader, v_loader = FashionMNIST(128, 64, sf=True)
    # ic(len(tset))
    # ic(len(vset))
    # ic(tset[0][0])

    # Test MNIST-by-label
    mnist = MNIST_By_CLASS()
    # for label in np.arange(10):
    #     ic(len(mnist[label][0]))

    # Test form ind dsets
    dset = form_ind_dsets(mnist, [0, 2, 3, 6, 8, 9])
    ic(len(dset))

    # Test sample from ood class
    # ood = sample_from_ood_class(mnist, [1,7], 32)
    # ic(len(ood))
    # dset = relabel_tuples(dset, [0, 2, 3, 6, 8, 9], np.arange(6))
    # check_classes(dset)
