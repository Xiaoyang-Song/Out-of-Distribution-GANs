from cProfile import label
from matplotlib import markers
from config import *
# Local saving address (may vary on different machines)
GD_LOSS_PLOT_ADDR = "../GDLossTracker/Examples/"
GD_LOSS_LOG_ADDR = "../GDLossTracker/"


class GD(Enum):
    G, D = list(range(2))


class Logger():
    # TODO: Wrap this with logging functions into a new logger class
    """
    This class is constructed for general debugging and logging purposes for OOD GANs.
    """

    def __init__(self, max_len=1000):
        """
        Note that in the below declaration:
        - 'n' stands for 'negative'
        - 'd' and 'g' stands for 'discriminator' amd 'generator', respectively
        - 'zsl' stands for "zero_softmax loss"
        - 'dist' stands for distances
        """
        # General Info
        self.max_len = max_len
        # Discriminator Loss
        self.d_zsl_ood = []
        self.d_zsl_fake = []
        self.d_ind_ce = []
        self.d_total = None
        # Generator Loss
        self.g_n_zsl_fake = []
        self.g_n_dist_fake_ind = []
        self.g_dist_fake_ood = []
        self.g_total = None

    def sum_up_gd_ls(self):
        self.d_total = np.array(self.d_zsl_fake) + \
            np.array(self.d_zsl_ood) + np.array(self.d_ind_ce)
        self.g_total = np.array(
            self.g_dist_fake_ood) + np.array(self.g_n_dist_fake_ind) + np.array(self.g_n_zsl_fake)
        ic('The total loss of G and D can be accessed now.')

    def ap_d_ls(self, ind_ce_loss, zsl_ood, zsl_fake):
        self.d_zsl_ood.append(zsl_ood.detach())
        self.d_zsl_fake.append(zsl_fake.detach())
        self.d_ind_ce.append(ind_ce_loss.detach())

    def ap_g_ls(self, zsl_fake, dist_fake_ind, dist_fake_ood):
        self.g_n_zsl_fake.append(zsl_fake.detach())
        self.g_n_dist_fake_ind.append(dist_fake_ind.detach())
        self.g_dist_fake_ood.append(dist_fake_ood.detach())

    def plt_ls(self, save_fname: str, num_iter: int, type: GD, verbose=False):
        # TODO: Change the implementation of this function to make it more elegan later.
        """
        Plot all three terms in the discriminator loss function

        Args:
        - save_fname: the saving destination of those plots.
        - num_iter: the number of iterations that are needed to be tracked.
        - type: either GD.G or GD.D, determines which to plot.
        - verbose (bool, optional): If verbose is True, plot them separately; otherwise,
          plot them in one plot. Defaults to False.
        """
        assert num_iter <= self.max_len, 'Expect num_iter to be less or equal than self.max_len.'
        x_axis = np.arange(num_iter)
        self.sum_up_gd_ls()
        if type == GD.D:
            # plt.plot(x_axis, self.d_zsl_fake[0:num_iter],
            #          marker='s', label='d_zsl_fake_ls')
            # plt.plot(x_axis, self.d_zsl_ood[0:num_iter],
            #          marker='+', label='d_zsl_ood_ls')
            # plt.plot(x_axis, self.d_ind_ce[0:num_iter],
            #          marker='x', label='d_ind_ce_ls')
            plt.plot(
                x_axis, self.d_zsl_fake[0:num_iter], label='W_z')
            plt.plot(
                x_axis, self.d_zsl_ood[0:num_iter], label='W_ood')
            plt.plot(x_axis, self.d_ind_ce[0:num_iter], label='CE')
            # plt.plot(x_axis, self.d_total[0:num_iter],
            #          marker='^', label='d_total_ls')
            plt.legend()
            # TODO: This implementation is silly, change this later.
            # plt.figure(dpi=100)
            plt.xlabel("Number of iterations")
            plt.title("Discriminator Loss vs. Number of iterations")
            plt.savefig(GD_LOSS_PLOT_ADDR + save_fname + f'[{num_iter}].jpg')
            plt.show()
            plt.close()
        elif type == GD.G:
            # plt.plot(x_axis, self.g_n_zsl_fake[0:num_iter],
            #          marker='o', label='g_n_zsl_fake_ls')
            # plt.plot(x_axis, self.g_n_dist_fake_ind[0:num_iter],
            #          marker='o', label='g_n_dist_fake_ind_ls')
            # plt.plot(x_axis, self.g_dist_fake_ood[0:num_iter],
            #          marker='o', label='g_dist_fake_ood_ls')
            plt.plot(
                x_axis, self.g_n_zsl_fake[0:num_iter], label='W_z')
            plt.plot(
                x_axis, self.g_n_dist_fake_ind[0:num_iter], label='d_Ind')
            # plt.plot(
            #     x_axis, self.g_dist_fake_ood[0:num_iter], label='d_ood^-')
            # plt.plot(x_axis, self.g_total[0:num_iter],
            #  marker='^', label='g_total_ls')
            plt.legend()
            # plt.figure(dpi=100)
            plt.xlabel("Number of iterations")
            plt.title("Generator Loss vs. Number of iterations")
            plt.savefig(GD_LOSS_PLOT_ADDR + save_fname + f'[{num_iter}].jpg')
            plt.show()
            plt.close()
        else:
            assert False, 'Unrecognized GD type.'
        if verbose:
            ic("Will be implemented soon.")
            return


def log_gd_loss(out_filename: str, num_iter: int, ls1, ls2, ls3, total_ls, type: str):
    # TODO: Implement this function and complete docstrings.
    """
    Log training loss into files.

    Args:
        out_filename (str): name of the destination logging file
        ls1 (_type_): _description_
        ls2 (_type_): _description_
        ls3 (_type_): _description_
        total_ls (_type_): _description_
        type (str): _description_
    """
    pass
# DISTANCE METRICS FOR OOD GANs


class DIST_TYPE(Enum):
    COR = ('Correlation', 0)
    EUC = ('Euclidean', 1)
    COS = ('Cosine Similarity', 2)


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
        return torch.abs(torch.corrcoef(target_mat)[0][1])
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
        assert False, 'Unexpected metric type.'
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
    # img_b1 = torch.rand((3, 28, 28))
    # img_b2 = torch.rand((4, 28, 28))
    # ic(get_dist_metric(img_b1, img_b2, 3, DIST_TYPE.COS))
    # for sample in DIST_TYPE:
    #     ic(sample)
    #     ic(sample == DIST_TYPE.COR)

    # Test show images
    # sample_imgs = torch.zeros((16, 16, 16))
    # sample_imgs = torch.ones((16, 16, 16)) * 0.1
    # show_images(sample_imgs)
    # plt.show()
