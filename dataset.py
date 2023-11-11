from random import sample
from config import *
# Auxiliary imports
from utils import visualize_img
from collections import defaultdict, Counter
import torchvision.transforms as trn
from image_folder import ImageSubfolder
from imagenet_loader import *

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


def PLACES365(bsz_tri, bsz_val, shuffle=True):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_data = torchvision.datasets.Places365(root="../Datasets/Places365", download=True, split='train-standard',
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    test_data = torchvision.datasets.Places365(root="../Datasets/Places365", split='val',download=True,
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
    tri_ldr = torch.utils.data.DataLoader(train_data, batch_size=bsz_tri, shuffle=True,
                                            num_workers=2, pin_memory=True)
    val_ldr = torch.utils.data.DataLoader(test_data, batch_size=bsz_val, shuffle=True,
                                            num_workers=2, pin_memory=True)
    return train_data, test_data, tri_ldr, val_ldr

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
    def __init__(self, dset_name, is_within_dset, bsz_tri, bsz_val, ind=None, ood=None):
        self.within_dset = is_within_dset
        self.name = dset_name
        self.bsz_tri = bsz_tri
        self.bsz_val = bsz_val
        self.ind, self.ood = ind, ood
        self.initialize()

    def initialize(self):
        if self.name in ['MNIST', 'FashionMNIST', 'SVHN']:

            assert self.ind is not None and self.ood is not None
            if self.name == 'MNIST':
                dset_tri, dset_val, _, _ = MNIST(self.bsz_tri, self.bsz_val)

            elif self.name == "SVHN":
                dset_tri, dset_val, _, _ = SVHN(self.bsz_tri, self.bsz_val)

            else:
                dset_tri, dset_val, _, _ = FashionMNIST(
                    self.bsz_tri, self.bsz_val, True)
            self.train = dset_by_class(dset_tri)
            self.val = dset_by_class(dset_val)
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
            self.ood_val_loader = set_to_loader(
                self.ood_val, self.bsz_val, True)

        elif self.name == 'MNIST-FashionMNIST':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = MNIST(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, self.ood_val_loader = FashionMNIST(
                self.bsz_tri, self.bsz_val, True)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        elif self.name == 'FashionMNIST-MNIST':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = FashionMNIST(
                self.bsz_tri, self.bsz_val, True)
            self.ood_train, self.ood_val, _, self.ood_val_loader = MNIST(
                self.bsz_tri, self.bsz_val)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        elif self.name == 'CIFAR10-SVHN':
            self.ind_train, self.ind_val, self.ind_train_loader, self.ind_val_loader = CIFAR10(
                self.bsz_tri, self.bsz_val)
            self.ood_train, self.ood_val, _, self.ood_val_loader = SVHN(
                self.bsz_tri, self.bsz_val)
            self.ood_train_by_class = dset_by_class(
                self.ood_train)  # this is used for sampling

        else:
            assert False, 'Unrecognized Dataset Combination.'

    def ood_sample(self, n, regime, idx=None):
        dset = self.train if self.within_dset else self.ood_train_by_class
        cls_lst = np.array(self.ood) if self.within_dset else np.arange(10)
        if regime == 'Balanced':
            idx_lst = cls_lst
        elif regime == 'Imbalanced':
            assert idx is not None
            idx_lst = cls_lst[idx]
        else:
            assert False, 'Unrecognized Experiment Type.'
        ood_sample = sample_from_ood_class(dset, idx_lst, n)
        ood_img_batch, ood_img_label = tuple_list_to_tensor(ood_sample)
        return ood_img_batch, ood_img_label


def line(n=80):
    return "="*n

def set_loader(bsz_tri, bsz_val):
    train_transform = trn.Compose([
        trn.Resize(size=224, interpolation=trn.InterpolationMode.BICUBIC),
        # trn.RandomResizedCrop(size=(224, 224), scale=(0.5, 1), interpolation=trn.InterpolationMode.BICUBIC),
        trn.RandomHorizontalFlip(p=0.5),
        trn.ToTensor(),
        trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    test_transform = trn.Compose([
        trn.Resize(size=(224, 224), interpolation=trn.InterpolationMode.BICUBIC),
        trn.CenterCrop(size=(224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    root_dir = '/nobackup-slow/dataset/ILSVRC-2012/'
    train_dir = root_dir + 'val'
    classes, _ = torchvision.datasets.folder.find_classes(train_dir)
    index = [125, 788, 630, 535, 474, 694, 146, 914, 447, 208, 182, 621, 271, 646, 328, 119, 772, 928, 610, 891, 340,
             890, 589, 524, 172, 453, 869, 556, 168, 982, 942, 874, 787, 320, 457, 127, 814, 358, 604, 634, 898, 388,
             618, 306, 150, 508, 702, 323, 822, 63, 445, 927, 266, 298, 255, 44, 207, 151, 666, 868, 992, 843, 436, 131,
             384, 908, 278, 169, 294, 428, 60, 472, 778, 304, 76, 289, 199, 152, 584, 510, 825, 236, 395, 762, 917, 573,
             949, 696, 977, 401, 583, 10, 562, 738, 416, 637, 973, 359, 52, 708]

    num_classes = 100
    classes = [classes[i] for i in index]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_data = ImageSubfolder(root_dir + 'train', transform=test_transform, class_to_idx=class_to_idx)
    test_data = ImageSubfolder(root_dir + 'val', transform=test_transform, class_to_idx=class_to_idx)
    labeled_trainloader = torch.utils.data.DataLoader(train_data, \
                                                      batch_size=bsz_tri, shuffle=True, num_workers=16,
                                                      pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=bsz_val, shuffle=True, num_workers=16, pin_memory=True)
    return labeled_trainloader, testloader


if __name__ == '__main__':
    # labeled_trainloader, testloader = set_loader(256, 256)
    train_data, test_data, tri_ldr, val_ldr = PLACES365(128, 128)