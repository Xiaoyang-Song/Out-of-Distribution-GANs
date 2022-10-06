from config import *
from dataset import *
import umap
import umap.plot
from sklearn.datasets import load_digits


def umap_visualization(x, y, color_key):
    mapper = umap.UMAP().fit(x)
    p = umap.plot.points(mapper, labels=y, color_key=color_key)
    umap.plot.show(p)


if __name__ == "__main__":
    ic("umap.py")
    digits = load_digits()
    umap_visualization(digits.data, digits.target)
