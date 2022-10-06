from config import *
from Visualization.umap_plot import umap_visualization
from scipy.stats import qmc


def sobol_seq(d: int, m: int, scramble: bool):
    sampler = qmc.Sobol(d=d, scramble=scramble)
    sample = sampler.random_base2(m=m)
    return torch.tensor(sample)


if __name__ == '__main__':
    ic("Hello sobol_seq.py")
    sobol_img = sobol_seq(d=784, m=10, scramble=False)
    umap_visualization(sobol_img, torch.ones(2**10))
