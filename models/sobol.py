from config import *
from sobol_seq import sobol_seq


def knn_search(Decoder, ood_feat_vec, k, d, m):
    # ood_feat_vec: B x d
    assert len(ood_feat_vec.shape) == 2
    assert ood_feat_vec.shape[1] == d
    # sobol_feat: 2^m x d
    sobol_feat = sobol_seq(d=d, m=m, scramble=False)
    # Matrix Broadcasting
    


if __name__ == '__main__':
    ic('Sobol sequence generation')
