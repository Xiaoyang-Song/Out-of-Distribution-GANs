from config import *
from dataset import *
import umap
import umap.plot
# from umap import plotpip install umap-learn[plot]
from sklearn.datasets import load_digits


if __name__ == "__main__":
    ic("umap.py")
    digits = load_digits()

    mapper = umap.UMAP().fit(digits.data)
    umap.plot.points(mapper, labels=digits.target)