from config import *
from dataset import *


class BinaryDataset(Dataset):
    def __init__(self, g_img, ind_img):
        assert type(g_img) == torch.tensor
        assert type(ind_img) == torch.tensor
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


def get_label_ind(ind_tri_set, size):
    pass


def get_binary_dataset(adv_img, ind_img):
    pass


def detector_trainer(dset, B_tri, B_val):
    pass


if __name__ == '__main__':
    ic("Hello detector_trainer.py")
    idx_ind = [0, 1, 3, 4, 5]
    dset_dict = MNIST_SUB(batch_size=2, val_batch_size=64,
                          idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    tri_set = dset_dict['train_set_ind']
    print(len(tri_set))
    ic(torch.cat([xy[0] for xy in tri_set]).shape)

    # cifar, _, _, _ = CIFAR10(64,64)
    # ic(len(cifar))
    # ic(cifar[0][0].shape)

    g_img = torch.load("checkpoint/adv_g_img.pt")
    ic(g_img.shape)