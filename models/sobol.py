from config import *
from sobol_seq import sobol_seq


def knn_search(Decoder, ood_feat_vec, k, d, m):
    # ood_feat_vec: B x d
    assert len(ood_feat_vec.shape) == 2
    assert ood_feat_vec.shape[1] == d
    # sobol_feat: 2^m x d
    sobol_feat = sobol_seq(d=d, m=m, scramble=False)
    # Matrix Broadcasting: B x 2^m
    dist = (ood_feat_vec - sobol_feat.T).norm(dim=1)
    # idx: B x k
    vals, idx = torch.topk(dist, dim=1, k=k, largest=False)
    idx = idx.squeeze()
    # Bk x d
    feat_vec = torch.cat([sobol_feat[i.squeeze()] for i in idx]).squeeze()
    # Generate samples
    return feat_vec, Decoder(feat_vec)


if __name__ == '__main__':
    ic('Sobol sequence generation')
