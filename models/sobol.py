from config import *
from sobol_seq import sobol_seq


def knn_search(Decoder, ood_feat_vec, k, d, num_grids):
    # ood_feat_vec: B x d
    assert len(ood_feat_vec.shape) == 2
    assert ood_feat_vec.shape[1] == d

    sobol_feat = sobol_seq(d=d, m=num_grids, scramble=False)
    

if __name__ == '__main__':
    ic('Sobol sequence generation')
