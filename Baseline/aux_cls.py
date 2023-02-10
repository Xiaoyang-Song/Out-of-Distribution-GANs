from dataset import *
from config import *
from models.mnist_cnn import MNISTCNN_AUX
from trainers.trainer import train


def MNIST_aux_dset(ind, ood, b_ood):
    mnist_t, mnist_v = MNIST_By_CLASS(train=True), MNIST_By_CLASS(train=False)
    xin_t, xout_t = form_ind_dsets(mnist_t, ind), form_ind_dsets(mnist_t, ood)
    xin_v, xout_v = form_ind_dsets(mnist_v, ind), form_ind_dsets(mnist_v, ood)
    ic(f"xin_t: {len(xin_t)}; xout_t: {len(xout_t)}")
    ic(f"xin_v: {len(xin_v)}; xout_v: {len(xout_v)}")

    # Form training dataset
    xin_t = relabel_tuples(xin_t, ind, np.arange(len(ind)))
    ood_dset = sample_from_ood_class(mnist_t, ood, b_ood)
    ood_dset_t = relabel_tuples(ood_dset, ood, [len(ind)]*len(ood))
    # dset_t = xin_t + ood_dset_t
    # check_classes(dset_t)
    # train_loader = set_to_loader(dset_t, bsz_tri, True)

    # Form unseen validation/testing dataset
    xin_v = relabel_tuples(xin_v, ind, np.arange(len(ind)))
    xout_v = relabel_tuples(xout_v, ood, [len(ind)]*len(ood))
    # dset_v = xin_v + xout_v
    # check_classes(dset_v)
    # val_loader = set_to_loader(dset_v, bsz_val, True)

    return xin_t, ood_dset_t, xin_v, xout_v


ind = [2, 3, 6, 8, 9]
ood = [1, 7]
xin_t, ood_dset_t, xin_v, xout_v = MNIST_aux_dset(ind, ood, 32)
dset_t = xin_t + ood_dset_t
dset_v = xin_v + xout_v

bsz_tri = 128
bsz_val = 128

train_loader = set_to_loader(dset_t, bsz_tri, True)
val_loader = set_to_loader(dset_v, bsz_val, True)

model = MNISTCNN_AUX(len(ind))
train(model, train_loader, val_loader, 8)

# Evaluation on testing set
xin, yin = tuple_list_to_tensor(xin_v)
xout, yout = tuple_list_to_tensor(xout_v)

pred = model(xin)

