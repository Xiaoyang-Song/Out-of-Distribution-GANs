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


def get_acc(model, dset):
    xin, yin = tuple_list_to_tensor(dset)
    logits = model(xin)
    acc = (torch.argmax(logits, dim=1) == yin).sum().item() / yin.shape[0]
    return acc


ind = [2, 3, 6, 8, 9]
ood = [1, 7]

# ood_bsz_lst = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
ood_bsz_lst = [4, 8]
ind_acc_lst, ood_acc_lst = [], []
num_epoch = 4
bsz_tri = 128
bsz_val = 128

for ood_bsz in ood_bsz_lst:
    # ood_bsz = 32
    xin_t, ood_dset_t, xin_v, xout_v = MNIST_aux_dset(ind, ood, ood_bsz)
    dset_t = xin_t + ood_dset_t
    dset_v = xin_v + xout_v

    train_loader = set_to_loader(dset_t, bsz_tri, True)
    val_loader = set_to_loader(dset_v, bsz_val, True)

    model = MNISTCNN_AUX(len(ind))
    model = train(model, train_loader, val_loader, num_epoch)

    # Evaluation on testing set
    print(f"B_OoD: {ood_bsz}")
    ind_acc, ood_acc = get_acc(model, xin_v), get_acc(model, xout_v)
    print(f"InD Acc: {ind_acc}")
    print(f"OoD Acc: {ood_acc}")

    ind_acc_lst.append(ind_acc)
    ood_acc_lst.append(ood_acc)

ic(ind_acc_lst)
ic(ood_acc_lst)


plt.plot(ood_bsz_lst, ind_acc_lst, marker='o', label="InD")
plt.plot(ood_bsz_lst, ood_acc_lst, marker='s', label="OoD")
# plt.plot(x, (y_m + y_fm) / 2, marker='x', label="Total")
plt.legend()
# plt.xlim(xmin=10)
# plt.ylim(ymin=0.1, ymax=0.7)
plt.xlabel("Number of OoD Samples Seen")
plt.xticks(ood_bsz_lst, ood_bsz_lst)
plt.title("Detection Accuracy vs. Number of OoD Samples")
plt.savefig("checkpoint/Baseline/mnist_4_8.png")
plt.close()
