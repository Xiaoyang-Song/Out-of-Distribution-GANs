from config import *
from wasserstein import *
from trainers.detector_trainer import *


def img_to_wloss(D, img):
    assert type(img) == torch.Tensor or type(
        img) == list, 'Expect input to be either a torch tensor or a list of tuples'
    if type(img) == torch.Tensor:
        w_loss = np.array([ood_wass_loss(D(x)).item() for x in img])
    else:
        w_loss = [ood_wass_loss(D(x[0])) for x in img]
    return np.array(w_loss)


def raw_to_nll(w_loss):
    return -np.log(w_loss)


def plot_w_loss(w_loss: list, color_key: list, path: str):
    # w_loss is a list of list
    assert len(w_loss) == len(color_key)
    for idx, wass in enumerate(w_loss):
        num_pts = len(wass)
        noise_y = np.random.normal(0, 1, num_pts)
        plt.scatter(wass, noise_y, c=color_key[idx])
    plt.savefig(path)


if __name__ == '__main__':
    ic("Hello wloss_visualization.py")
    idx_ind = [0, 1, 3, 4, 5]
    dset_dict = MNIST_SUB(batch_size=2, val_batch_size=64,
                          idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    tri_set = dset_dict['train_set_ind'][0:100]
    ood_set = dset_dict['train_set_ood'][0:100]

    D = dc_discriminator(GAN_TYPE.OOD).to(DEVICE)
    D.load_state_dict(torch.load("checkpoint/adv_pre_D.pt"))
    print("Pretrained D state is loaded.")

    g_img = torch.load("checkpoint/adv_g_img(cpu).pt")
    # Test visualization functionality
    color = ['#ffbf00', '#00b384']
    w_pts = [img_to_wloss(D, dset) for dset in (tri_set, ood_set)]
    plot_w_loss(w_pts, color, "Visualization/w_vis.png")
