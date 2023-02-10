from dataset import *
from config import *


def MNIST_baseline(ind, ood, b_ood):
    mnist = MNIST_By_CLASS()
    ind_dset = form_ind_dsets(mnist, ind)
    ind_dset = relabel_tuples(ind_dset, ind, np.arange(len(ind)))
    ood_dset = sample_from_ood_class(mnist, ood, b_ood)
    ood_dset = relabel_tuples(ood_dset, ood, [len(ind)]*len(ood))
    dset = ind_dset + ood_dset
    check_classes(dset)


# MNIST_baseline([2,3,6,8,9], [1,7], 32)
