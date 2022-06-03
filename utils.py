from config import *


class GDLoss():
    """
    This class is constructed for general debugging and logging purposes for OOD GANs.
    """

    def __init__(self, max_len=1000):
        """
        Note that in the below declaration:
        - 'n' stands for 'negative'
        - 'd' and 'g' stands for 'discriminator' amd 'generator', respectively
        - 'zsl' stands for "zero_soft_max loss"
        - 'dist' stands for distances
        """
        # General Info'
        self.max_len = max_len
        # Discriminator Loss
        self.d_zsl_ood = []
        self.d_zsl_fake = []
        self.d_ind_ce = []
        # Generator Loss
        self.g_n_zsl_fake = []
        self.g_n_dist_fake_ind = []
        self.g_dist_fake_ood = []

    def ap_d_ls(self, ind_ce_loss, zsl_ood, zsl_fake):
        self.d_zsl_ood.append(zsl_ood)
        self.d_zsl_fake.append(zsl_fake)
        self.d_ind_ce.append(ind_ce_loss)

    def ap_g_ls(self, zsl_fake, dist_fake_ind, dist_fake_ood):
        self.g_n_zsl_fake.append(-zsl_fake)
        self.g_n_dist_fake_ind .append(dist_fake_ind)
        self.g_dist_fake_ood.append(dist_fake_ood)

    def plt_d_ls(self, save_addr, num_iter, verbose=False):
        """
        Plot all three terms in the discriminator loss function

        Args:
        - save_addr: the saving destination of those plots.
        - num_iter: the number of iterations that are needed to be tracked.
        - verbose (bool, optional): If verbose is True, plot them separately; otherwise,
          plot them in one plot. Defaults to False.
        """
        assert num_iter <= self.max_len, 'Expect num_iter to be less or equal than self.max_len.'
        x_axis = np.arange(num_iter)
        plt.plot()


class DIST_TYPE(Enum):
    COR = ('Correlation', 0)
    EUC = ('Euclidean', 1)
    COS = ('Cosine Similarity', 2)

# DISTANCE METRICS FOR OOD GANs


def get_dist_metric(img_b1: torch.Tensor, img_b2: torch.Tensor,
                    m: int, type: DIST_TYPE):
    """
    Calculate the distance between two batches of raw images
    Inputs:
    - img_b1: tensor of shape B1 x H x W
    - img_b2: tensor of shape B2 x H x W 
    - m: number of samples from both batches
    - type: distance metric type
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
    assert B1 >= m and B2 >= m, 'Expect the sample size less or equal than batch sizes.'
    # Sample images from both batches
    idx1, idx2 = torch.randint(0, B1, (m,)), torch.randint(0, B2, (m,))
    if len(img_b1.shape) == 3:
        img_b1_sub, img_b2_sub = img_b1[idx1, :, :], img_b2[idx2, :, :]
    else:
        img_b1_sub, img_b2_sub = img_b1[idx1, :], img_b2[idx2, :]
    # Compute distances
    # TODO: Change the following segments for efficiency and elegancy later.
    if type == DIST_TYPE.COR:
        # Compute sample mean of two sampled batch
        img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(1, -1)  # 1 x HW
        img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(1, -1)  # 1 x HW
        target_mat = torch.cat([img_b1_sub, img_b2_sub])  # 2 x HW
        return torch.corrcoef(target_mat)[0][1]
    elif type == DIST_TYPE.EUC:
        # Compute sample mean of two sampled batch
        img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(1, -1)  # 1 x HW
        img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(1, -1)  # 1 x HW
        l2_euc = torch.sqrt(torch.sum((img_b1_sub - img_b2_sub)**2))
        return l2_euc
    elif type == DIST_TYPE.COS:
        img_b1_sub = torch.mean(img_b1_sub, dim=0).reshape(-1,)  # HW
        img_b2_sub = torch.mean(img_b2_sub, dim=0).reshape(-1,)  # HW
        # Compute norm
        norm1, norm2 = LA.norm(img_b1_sub), LA.norm(img_b2_sub)
        cosine_sim = torch.dot(img_b1_sub, img_b2_sub) / (norm1 * norm2)
        return cosine_sim
    else:
        return None

# VISUALIZATION UTILITY FUNCTIONS


def visualize_img(loader: DataLoader):
    # TODO: Finish implementing this utility function and incoporate it in the notebook
    pass


def show_images(images):
    images = torch.reshape(
        images, [images.shape[0], -1]
    )  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


if __name__ == '__main__':
    ic("Hello utils.py")
    # TEST get_dist_metric()
    img_b1 = torch.rand((3, 28, 28))
    img_b2 = torch.rand((4, 28, 28))
    ic(get_dist_metric(img_b1, img_b2, 3, DIST_TYPE.COS))
