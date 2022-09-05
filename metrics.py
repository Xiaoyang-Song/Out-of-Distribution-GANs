from config import *
from utils import DIST_TYPE
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models import resnet50, ResNet50_Weights


class CosSim():
    def __init__(self, model, weights, layer_to_delete):
        self.weights = weights.DEFAULT
        self.model = model(weights=self.weights)
        self.delete = layer_to_delete
        nodes, _ = get_graph_node_names(model)
        self.preprocess = weights.transforms()
        self.feat_extractor = create_feature_extractor(
            self.model, return_nodes=nodes[:-self.delete])
        self.feat_extractor.requires_grad_(False)

    def __call__(self, x, y):
        assert len(x.shape) == 4 and x.shape[1] == 3, 'Wrong Tensor Dims'
        assert len(y.shape) == 4 and y.shape[1] == 3, 'Wrong Tensor Dims'
        pre_x, pre_y = [self.preprocess(input) for input in (x, y)]
        def forward(x): return self.feat_extractor(x).squeeze().mean(dim=0)
        out_x, out_y = [forward(raw) for raw in (pre_x, pre_y)]
        normx, normy = [LA.norm(target) for target in (out_x, out_y)]
        return torch.dot(out_x, out_y) / (normx * normy)


class Metric():
    # TODO: Finish implementing this class
    def __init__(self, dist_type: DIST_TYPE, m: int):
        self.dist_type = dist_type
        self.sample_size = m

    def __init__(self):
        pass

    def get_dist(self, img_b1: torch.Tensor, img_b2: torch.Tensor):
        """
        Calculate the distance between two batches of raw images
        Inputs:
        - img_b1: tensor of shape B1 x H x W
        - img_b2: tensor of shape B2 x H x W
        Returns:
        - A torch.float32 value that gives the similarity between two images.
        """
        # Assertion check
        assert len(img_b1.shape) == len(
            img_b2.shape), 'Two input batches should have the same number of dimensions.'
        # assert len(img_b1.shape) == 3, 'Expect input tensors have shape B x H x W.'
        if len(img_b1.shape) == 3:
            B1, H, W = img_b1.shape
            B2, _, _ = img_b2.shape
        elif len(img_b2.shape) == 2:
            B1, HW = img_b1.shape
            B2, HW = img_b2.shape
        else:
            assert False, 'Expected image batch to have shape of either (B, H, W) or (B, HW)'
        assert B1 >= self.m and B2 >= self.m, 'Expect the sample size less or equal than batch sizes.'
        # Sample images from both batches
        idx1, idx2 = torch.randint(
            0, B1, (self.m,)), torch.randint(0, B2, (self.m,))
        if len(img_b1.shape) == 3:
            img_b1_sub, img_b2_sub = img_b1[idx1, :, :], img_b2[idx2, :, :]
        else:
            img_b1_sub, img_b2_sub = img_b1[idx1, :], img_b2[idx2, :]
        # Compute distances
        # TODO: Change the following segments for efficiency and elegancy later.
        if self.dist_type == DIST_TYPE.COR:
            # Compute sample mean of two sampled batch
            img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(1, -1)  # 1 x HW
            img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(1, -1)  # 1 x HW
            target_mat = torch.cat([img_b1_sub, img_b2_sub])  # 2 x HW
            return torch.corrcoef(target_mat)[0][1]
        elif self.dist_type == DIST_TYPE.EUC:
            # Compute sample mean of two sampled batch
            img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(1, -1)  # 1 x HW
            img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(1, -1)  # 1 x HW
            l2_euc = torch.sqrt(torch.sum((img_b1_sub - img_b2_sub)**2))
            return l2_euc
        elif self.dist_type == DIST_TYPE.COS:
            img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(-1,)  # HW
            img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(-1,)  # HW
            # Compute norm
            norm1, norm2 = LA.norm(img_b1_sub), LA.norm(img_b2_sub)
            cosine_sim = torch.dot(img_b1_sub, img_b2_sub) / (norm1 * norm2)
            return cosine_sim
        else:
            return None
