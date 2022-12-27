from config import *
from dataset import *

def nn_search(k, sample_size):
    pass

if __name__ == '__main__':
    ic("Boundary points processing...")

    # MNIST experiments
    ind, ood = [2, 3, 6, 8, 9], [1, 7]
    mnist = MNIST_By_CLASS()
    xin, xout = form_ind_dsets(mnist, ind), form_ind_dsets(mnist, ood)
    ic(f"xin: {len(xin)}; xout: {len(xout)}")
    # Cast to tensors for KNN search
    xin, yin = tuple_list_to_tensor(xin)
    ic(f"{xin.shape}, {yin.shape}")
    xout, yout = tuple_list_to_tensor(xout)
    ic(f"{xout.shape}, {yout.shape}")
