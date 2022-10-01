from config import *
from dataset import *


class BinaryDataset(Dataset):
    def __init__(self, g_img, ind_img, bs_t=64, bs_v=64, sample_ratio=1):
        assert type(g_img) == torch.Tensor
        # g_img should be a tensor of shape B x C x H x W
        self.g_img = g_img
        self.n_g = g_img.shape[0]
        # ind_img should be a list of tuple, which is what pytorch returns
        self.ind_img = torch.stack([x[0] for x in ind_img], dim=0)
        self.ind_y = torch.tensor([x[1] for x in ind_img])
        self.ind_cls = torch.unique(self.ind_y)
        self.n_ind = self.ind_img.shape[0]
        # Random sample ind data and get summary
        self.R = sample_ratio
        self.sample()
        self.info()
        ic(self.g_img.shape)
        ic(self.ind_img.shape)
        # Combine data and assign labels
        self.assign_label()

    def assign_label(self):
        # Data should be a list of tuples
        # No need to shuffle, Dataloader will do that for us.
        label = torch.ones(self.n_g * (self.R + 1))
        label[0:self.n_g] = 0
        self.g_ind_img = torch.cat([self.g_img, self.ind_img])
        self.data = [(label[i], self.g_ind_img[i])
                     for i in range(self.__len__())]

    def sample(self):
        rand_idx = np.random.choice(
            self.n_ind, self.n_g * self.R, replace=False)
        self.ind_img = self.ind_img[rand_idx]
        self.ind_y = self.ind_y[rand_idx]
        assert len(self.ind_y) == self.n_g * self.R
        ic(f"Sampling of In-Distribution data Done.")

    def info(self):
        ic(f"{self.ind_img.shape[0]} InD samples")
        ic(f"{self.n_g} OoD samples.")
        for idx in self.ind_cls:
            num = len(self.ind_y[self.ind_y == idx])
            ic(f"{idx}: {num} {num / self.n_g * self.R:.0%}")

    def __len__(self):
        return self.n_g * (self.R + 1)

    def __getitem__(self, index):
        assert index < self.__len__()
        return self.data[index]


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
    print(tri_set[0][0].shape)
    ind_set = torch.stack([xy[0] for xy in tri_set], dim=0)
    ind_y = torch.tensor([xy[1] for xy in tri_set])
    ic(ind_y.shape)
    ic(ind_set.shape)
    ic(ind_y)
    rand_idx = np.random.choice(30059, 1280, replace=False)
    print(rand_idx)
    sample_ind, sample_y = ind_set[rand_idx], ind_y[rand_idx]
    for x in idx_ind:
        ic(f"{x}: {len(sample_y[sample_y == x])}")
    # cifar, _, _, _ = CIFAR10(64,64)
    # ic(len(cifar))
    # ic(cifar[0][0].shape)
    ic(torch.unique(ind_y))
    g_img = torch.load("checkpoint/adv_g_img(cpu).pt")
    ic(type(g_img))
    ic(g_img.shape)
    data = BinaryDataset(g_img, tri_set, bs_t=64, bs_v=64, sample_ratio=1)
