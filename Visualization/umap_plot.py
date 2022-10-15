from config import *
from dataset import *
import umap
import umap.plot
from sklearn.datasets import load_digits


def umap_visualization(x, y, color_key, filename=None):
    mapper = umap.UMAP().fit(x)
    p = umap.plot.points(mapper, labels=y, color_key=color_key)
    umap.plot.show(p)


def mnist_fashion_mnist():
    mnist, _, _, _ = MNIST(
        128, 64, 2, True)
    mnist_img = torch.stack([x[0] / torch.sum(x[0])
                             for x in mnist]).flatten(1, 3)
    mnist_label = torch.ones(mnist_img.shape[0])

    fashionmnist, _, _, _ = FashionMNIST(128, 64, shuffle=True)
    fashionmnist_img = torch.stack([x[0] / torch.sum(x[0])
                                    for x in fashionmnist]).flatten(1, 3)
    fashionmnist_label = torch.ones(fashionmnist_img.shape[0])
    x_vis = torch.cat([mnist_img, fashionmnist_img])
    y_vis = torch.cat([mnist_label, fashionmnist_label])
    ic(x_vis.shape)
    ic(y_vis.shape)
    umap_visualization(x_vis, y_vis, ['#00b384', 'lightgray'])


if __name__ == "__main__":
    ic("umap.py")
    # digits = load_digits()
    # umap_visualization(digits.data, digits.target)
    mnist_fashion_mnist()
