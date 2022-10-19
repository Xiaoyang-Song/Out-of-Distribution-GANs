from config import *
from Visualization.umap_plot import umap_visualization
from scipy.stats import qmc
from dataset import MNIST


def sobol_seq(d: int, m: int, scramble: bool):
    sampler = qmc.Sobol(d=d, scramble=scramble)
    sample = sampler.random_base2(m=m)
    return torch.tensor(sample)


def visualize_sobol_seq():
    pass


if __name__ == '__main__':
    ic("Hello sobol_seq.py")
    sobol_img = sobol_seq(d=2, m=5, scramble=False)
    ic(sobol_img.shape)
    # sobol_y = torch.zeros(sobol_img.shape[0])
    ic(sobol_img)
    # umap_visualization(sobol_img, torch.ones(2**10))
    # ic(sobol_img[0])
    plt.scatter(sobol_img[:,0], sobol_img[:,1])
    plt.show()
    # mnist_tri_set, mnist_val_set, mnist_tri_loader, mnist_val_loader = MNIST(
    #     128, 64, 2, True)
    # mnist_tri_img = torch.stack([x[0] / torch.sum(x[0])
    #                             for x in mnist_tri_set]).flatten(1, 3)
    # mnist_tri_label = torch.ones(mnist_tri_img.shape[0])
    # ic(mnist_tri_img.shape)
    # # ic(torch.mean(mnist_tri_img))
    # x_vis = torch.cat([mnist_tri_img, sobol_img])
    # y_vis = torch.cat([mnist_tri_label, sobol_y])
    # ic(x_vis.shape)
    # ic(y_vis.shape)
    # umap_visualization(x_vis, y_vis, ['#00b384', 'lightgray'])
