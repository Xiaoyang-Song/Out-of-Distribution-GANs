from config import *
from dataset import *


def nn_search(xin, xout, k):
    xin, yin = tuple_list_to_tensor(xin)
    x, y = xin, yin
    # Find K Nearest Neighbor
    xin = xin.reshape((-1, 784, 1))
    xout = xout.reshape((-1, 784, 1))
    # ic(f"{xin.shape}; {xout.shape}; {xin.permute((2, 1, 0)).shape}")
    diff = xout - xin.permute((2, 1, 0))
    # ic(diff.shape)
    # Compute norm
    dist = torch.norm(diff, dim=1)
    ic(dist.shape)
    # K NN
    vals, idx = torch.topk(dist, dim=1, k=k, largest=False)
    idx = idx.squeeze()
    # ic(idx.shape)
    # Flatten and remove duplicate
    idx = idx.reshape((1, -1)).squeeze().unique()
    # ic(idx.shape)
    # Get results
    x, y = x[idx, :, :, :], y[idx]
    ic(f"{x.shape}; {y.shape}")
    return x, y


if __name__ == '__main__':
    ic("Boundary points processing...")

    # MNIST experiments
    ind, ood = [2, 3, 6, 8, 9], [1, 7]
    mnist = MNIST_By_CLASS()
    xin, xout = form_ind_dsets(mnist, ind), form_ind_dsets(mnist, ood)
    ic(f"xin: {len(xin)}; xout: {len(xout)}")
    xout, yout = tuple_list_to_tensor(xout)
    nn_search(xin, xout, 10)
    # Cast to tensors for KNN search
    # xin, yin = tuple_list_to_tensor(xin)
    # x, y = xin, yin
    # ic(f"{xin.shape}; {yin.shape}")
    # xout, yout = tuple_list_to_tensor(xout)
    # ic(f"{xout.shape}; {yout.shape}")
    # # Sample OOD points
    # num_ood = 32
    # idx = np.random.choice(len(xout), 32, False)
    # xout, yout = xout[idx, :, :, :], yout[idx]
    # ic(f"{xout.shape}; {yout.shape}")
    # # Find K Nearest Neighbor
    # xin = xin.reshape((-1, 784, 1))
    # xout = xout.reshape((-1, 784, 1))
    # ic(f"{xin.shape}; {xout.shape}; {xin.permute((2, 1, 0)).shape}")
    # diff = xout - xin.permute((2, 1, 0))
    # ic(diff.shape)
    # # Compute norm
    # dist = torch.norm(diff, dim=1)
    # ic(dist.shape)
    # # K NN
    # k = 10
    # vals, idx = torch.topk(dist, dim=1, k=k, largest=False)
    # idx = idx.squeeze()
    # ic(idx.shape)
    # # Flatten and remove duplicate
    # idx = idx.reshape((1, -1)).squeeze().unique()
    # ic(idx.shape)
    # # Get results
    # x, y = x[idx, :, :, :], y[idx]
    # ic(f"{x.shape}; {y.shape}")
