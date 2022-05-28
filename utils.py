from config import *


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
    assert len(img_b1.shape) == 3, 'Expect input tensors have shape B x H x W.'
    B1, H, W = img_b1.shape
    B2, _, _ = img_b2.shape
    assert B1 >= m and B2 >= m, 'Expect the sample size less or equal than batch sizes.'
    # Sample images from both batches
    idx1, idx2 = torch.randint(0, B1, (m,)), torch.randint(0, B2, (m,))
    img_b1_sub, img_b2_sub = img_b1[idx1, :, :], img_b2[idx2, :, :]
    # Compute distances
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
    ic(get_dist_metric(img_b1, img_b2, 3, DIST_TYPE.EUC))
