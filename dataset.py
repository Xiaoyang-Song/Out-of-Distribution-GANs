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


def dset_by_class(dset):
    ic(len(dset))
    img_lst = defaultdict(list)
    label_lst = defaultdict(list)
    # Loop through each tuple
    for item in dset:
        img_lst[item[1]].append(item[0])
        label_lst[item[1]].append(item[1])
    # Declare a wrapper dictionary
    dset_by_class = {}
    for label in np.arange(10):
        dset_by_class[label] = (img_lst[label], label_lst[label])
    return dset_by_class

# Specifically for Ind Ood Separation


def form_ind_dsets(input_dsets, ind_idx):
    dset = []
    for label in ind_idx:
        dset += list(zip(input_dsets[label][0], input_dsets[label][1]))
    return dset


def sample_from_ood_class(ood_dset: dict, ood_idx: list, sample_size):
    samples = []
    for idx in ood_idx:
        img, label = ood_dset[idx]
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
    def __init__(self, dset_name, bsz_tri, bsz_val, ind=None, ood=None):
        self.name = dset_name
        self.bsz_tri = bsz_tri
        self.bsz_val = bsz_val
        self.ind, self.ood = ind, ood
        self.initialize()

    def initialize(self):
        if self.name == 'MNIST':
            assert self.ind is not None and self.ood is not None
            mnist_tri, mnist_val, _, _ = MNIST(self.bsz_tri, self.bsz_val)
            self.train = dset_by_class(mnist_tri)
            self.val = dset_by_class(mnist_val)
            # The following code is for within-dataset InD/OoD separation
            self.ind_train = form_ind_dsets(self.train, self.ind)
            self.ind_val = form_ind_dsets(self.val, self.ind)
            self.ood_train = form_ind_dsets(self.train, self.ood)
            self.ood_val = form_ind_dsets(self.val, self.ood)
            self.ind_train = relabel_tuples(
                self.ind_train, self.ind, np.arange(len(self.ind)))
            self.ind_val = relabel_tuples(
                self.ind_val, self.ind, np.arange(len(self.ind)))
            self.ind_train_loader = set_to_loader(
                self.ind_train, self.bsz_tri, True)
            self.ind_val_loader = set_to_loader(
                self.ind_val, self.bsz_val, True)
        elif self.name == 'MNIST-FashionMNIST':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = MNIST(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, _ = FashionMNIST(
                self.bsz_tri, self.bsz_val, True)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling
        elif self.name == 'CIFAR10-SVHN':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR10(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, _ = SVHN(
                self.bsz_tri, self.bsz_val)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling
        else:
            assert False, 'Unrecognized Dataset Combination.'

    def ood_sample(self, n, regime, idx=None):
        dset = self.train if self.name == 'mnist' else self.ood_train_by_class
        cls_lst = self.ood if self.name == 'mnist' else np.arange(10)
        if regime == 'balanced':
            idx_lst = cls_lst
        elif regime == 'imbalanced':
            assert idx is not None
            idx_lst = cls_lst[idx]
        else:
            assert False, 'Unrecognized Experiment Type.'
        ood_sample = sample_from_ood_class(dset, idx_lst, n)
        ood_img_batch, ood_img_label = tuple_list_to_tensor(ood_sample)
        return ood_img_batch, ood_img_label


def line(n=40):
    return "-"*n


if __name__ == '__main__':
    # Test dataset functions
    dset = DSET('mnist-fashionmnist', 50, 128)
    ood_img_batch, ood_img_label = dset.ood_sample(2, 'imbalanced', [0])
    ic(ood_img_label)
    ic(ood_img_batch.shape)
    for img in ood_img_batch:
        plt.imshow(img.squeeze())
        plt.show()
