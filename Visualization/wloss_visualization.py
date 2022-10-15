from config import *
from wasserstein import *
from trainers.detector_trainer import *
from models.dc_gan_model import dc_discriminator, GAN_TYPE


def img_to_wloss(D, img):
    assert type(img) == torch.Tensor or type(
        img) == list, 'Expect input to be either a torch tensor or a list of tuples'
    if type(img) != torch.Tensor:
        img = torch.stack([x[0] for x in img], dim=0)
    # ic(D(img[0].unsqueeze(0)))
    w_loss = np.array(ood_wass_loss(F.softmax(D(img), dim=-1)))
    # ic(w_loss)
    return w_loss


def raw_to_nll(w_loss):
    # Filter out invalid wass_loss (floating point error)
    w_loss = w_loss[w_loss > 0]
    return -np.log(w_loss)


def plot_w_loss(w_loss: list, color_key: list, path: str):
    # w_loss is a list of list
    assert len(w_loss) == len(color_key)
    label = ['InD', 'OoD']
    for idx, wass in enumerate(w_loss):
        num_pts = len(wass)
        noise_y = np.random.normal(0, 1, num_pts)
        plt.scatter(wass, noise_y,
                    c=color_key[idx], label=label[idx], alpha=0.4)
    plt.legend()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    ic("Hello wloss_visualization.py")
    # idx_ind = [0, 1, 3, 4, 5]
    # dset_dict = MNIST_SUB(batch_size=2, val_batch_size=64,
    #                       idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    # tri_set = dset_dict['train_set_ind']
    # ood_set = dset_dict['train_set_ood']

    img_info = {'H': 28, 'W': 28, 'C': 1}
    # Load pretrained discriminator
    D = dc_discriminator(img_info, GAN_TYPE.OOD).to(DEVICE)
    path = 'checkpoint/adv_pre_D.pt'
    pretrain = torch.load(path)
    D.load_state_dict(pretrain['model_state_dict'])
    print("Pretrained D state is loaded.")

    # g_img = torch.load("checkpoint/adv_g_img(cpu).pt")
    # # Test visualization functionality
    # color = ['#ffbf00', '#00b384']
    # w_pts = [raw_to_nll(img_to_wloss(D, dset)) for dset in (tri_set, ood_set)]
    # plot_w_loss(w_pts, color, "Visualization/w_vis_ood_ind.png")
    # w_pts = [raw_to_nll(img_to_wloss(D, dset)) for dset in (tri_set, g_img)]
    # plot_w_loss(w_pts, color, "Visualization/w_vis_g_ind.png")

    # Visualize MNIST FASHIONMNIST W Loss
    