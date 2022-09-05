from config import *
from utils import DIST_TYPE
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import resnet50, ResNet50_Weights
from dataset import MNIST, CIFAR10, MNIST_SUB


class CosSim():
    def __init__(self, model, weights, layer_to_delete):
        self.weights = weights.DEFAULT
        self.model = model(weights=self.weights)
        self.delete = layer_to_delete
        nodes, _ = get_graph_node_names(self.model)
        # print(nodes)
        self.preprocess = self.weights.transforms()
        self.feat_extractor = create_feature_extractor(
            self.model, return_nodes=nodes[:-self.delete])
        self.map_idx = nodes[-self.delete-1]
        # print(self.map_idx)
        self.feat_extractor.requires_grad_(False)
        print('CosSim Metric Created.')

    def __call__(self, x, y):
        assert len(x.shape) == 4 and x.shape[1] == 1, 'Wrong Tensor Dims'
        assert len(y.shape) == 4 and y.shape[1] == 1, 'Wrong Tensor Dims'
        pre_x, pre_y = x, y
        # pre_x, pre_y = [self.preprocess(input) for input in (x, y)]
        # print(pre_x.shape)

        def forward(x): return self.feat_extractor(
            x)[self.map_idx].squeeze().mean(dim=0)
        out_x, out_y = [forward(raw) for raw in (pre_x, pre_y)]
        # print(out_x.shape)
        norm_x, norm_y = [LA.norm(target) for target in (out_x, out_y)]
        # print(norm_x.shape)
        return torch.dot(out_x, out_y) / (norm_x * norm_y)


if __name__ == '__main__':
    print('Hello metrics.py')
    metric = CosSim(resnet50, ResNet50_Weights, 2)
    ind_idx = [0, 2, 3, 6, 8]
    ood_idx = [1, 7]
    # TODO: Use classifier on 1,4,5,7,9, check wass loss
    # test_idx = [4,5,9]
    mnist_dset_dict = MNIST_SUB(
        batch_size=128, val_batch_size=64, idx_ind=ind_idx, idx_ood=ood_idx, shuffle=True)
    ind_train_loader = mnist_dset_dict['train_set_ind_loader']
    ood_train_loader = mnist_dset_dict['train_set_ood_loader']
    ind_val_loader = mnist_dset_dict['val_set_ind_loader']

    x0, y0 = next(iter(ind_train_loader))
    # Temporary test
    x0 = torch.repeat_interleave(x0, 3, 1)
    x1, y1 = next(iter(ind_val_loader))
    # Temporary test
    x1 = torch.repeat_interleave(x1, 3, 1)

    # 0.9569 for ind_train and ood_train
    # 0.9931 for ind_train and ind_val
    print(metric(x0, x1))

 #   class Metric():
    #     # TODO: Finish implementing this class
#     def __init__(self, dist_type: DIST_TYPE, m: int):
#         self.dist_type = dist_type
#         self.sample_size = m

#     def __init__(self):
#         pass

#     def get_dist(self, img_b1: torch.Tensor, img_b2: torch.Tensor):
#         """
#         Calculate the distance between two batches of raw images
#         Inputs:
#         - img_b1: tensor of shape B1 x H x W
#         - img_b2: tensor of shape B2 x H x W
#         Returns:
#         - A torch.float32 value that gives the similarity between two images.
#         """
#         # Assertion check
#         assert len(img_b1.shape) == len(
#             img_b2.shape), 'Two input batches should have the same number of dimensions.'
#         # assert len(img_b1.shape) == 3, 'Expect input tensors have shape B x H x W.'
#         if len(img_b1.shape) == 3:
#             B1, H, W = img_b1.shape
#             B2, _, _ = img_b2.shape
#         elif len(img_b2.shape) == 2:
#             B1, HW = img_b1.shape
#             B2, HW = img_b2.shape
#         else:
#             assert False, 'Expected image batch to have shape of either (B, H, W) or (B, HW)'
#         assert B1 >= self.m and B2 >= self.m, 'Expect the sample size less or equal than batch sizes.'
#         # Sample images from both batches
#         idx1, idx2 = torch.randint(
#             0, B1, (self.m,)), torch.randint(0, B2, (self.m,))
#         if len(img_b1.shape) == 3:
#             img_b1_sub, img_b2_sub = img_b1[idx1, :, :], img_b2[idx2, :, :]
#         else:
#             img_b1_sub, img_b2_sub = img_b1[idx1, :], img_b2[idx2, :]
#         # Compute distances
#         # TODO: Change the following segments for efficiency and elegancy later.
#         if self.dist_type == DIST_TYPE.COR:
#             # Compute sample mean of two sampled batch
#             img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(1, -1)  # 1 x HW
#             img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(1, -1)  # 1 x HW
#             target_mat = torch.cat([img_b1_sub, img_b2_sub])  # 2 x HW
#             return torch.corrcoef(target_mat)[0][1]
#         elif self.dist_type == DIST_TYPE.EUC:
#             # Compute sample mean of two sampled batch
#             img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(1, -1)  # 1 x HW
#             img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(1, -1)  # 1 x HW
#             l2_euc = torch.sqrt(torch.sum((img_b1_sub - img_b2_sub)**2))
#             return l2_euc
#         elif self.dist_type == DIST_TYPE.COS:
#             img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(-1,)  # HW
#             img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(-1,)  # HW
#             # Compute norm
#             norm1, norm2 = LA.norm(img_b1_sub), LA.norm(img_b2_sub)
#             cosine_sim = torch.dot(img_b1_sub, img_b2_sub) / (norm1 * norm2)
#             return cosine_sim
#         else:
#             return None
