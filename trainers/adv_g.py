from config import *
from wass_loss import *
from dataset import *
from utils import *
from trainers.ood_gan_trainer import *
from models.dc_gan_model import dc_discriminator, dc_generator
from wasserstein import Wasserstein
from Visualization.umap_plot import umap_visualization


def grad_asc_w_rej(ind_loader, D, B, M, Wt, WLoss, device=DEVICE):
    G_x = []
    for i, (x, y) in enumerate(ind_loader):
        for img in x:
            p = img.clone().unsqueeze(0).to(device)
            p.requires_grad_()  # Enforce requires grad
            g_x = ad_atk(D, p, Wt, WLoss)
            G_x.append(g_x)

        if i >= M:
            ic(f"{B * M} adversarial samples are generated.")
            return G_x
    return G_x


def ad_atk(D, x, Wt, WLoss):
    assert x.requires_grad
    ic(x.shape)
    x = nn.Parameter(x)
    optimizer = torch.optim.Adam([{"params": x}], lr=1e-3)
    i = 1
    while True:
        # Aim to minimize W
        optimizer.zero_grad()
        logit = D(x)
        # Minimize negative Wasserstein distances to onehot vectors
        W = -WLoss(torch.softmax(logit, dim=-1))
        # ic(W)
        # TODO: Need to handle the case where there is an explosion
        if W < Wt:
            print(f"Adversarial Attack done in {i} iterations")
            ic(torch.softmax(logit, dim=-1))
            return x.detach()
        W.backward()
        optimizer.step()
        i += 1


def to_raw(W_t):
    return -math.exp(-W_t)


def find_W_t():
    # TODO: Implement this function later
    return 1.5


def adv_generation(ind_loader, D, B, M, nl_Wt, device=DEVICE):
    WLoss = Wasserstein.apply
    Wt = to_raw(nl_Wt)
    G_x = grad_asc_w_rej(ind_loader, D, B, M, Wt, WLoss, device=device)
    adv_g_img = torch.cat(G_x)
    return adv_g_img


if __name__ == '__main__':
    ic("Hello adv_g.py")

    # Load pretrained_D
    img_info = {'H': 28, 'W': 28, 'C': 1}
    D = dc_discriminator(img_info, GAN_TYPE.OOD).to(DEVICE)
    path = 'checkpoint/pretrained_D_low.pt'
    pretrain = torch.load(path)
    D.load_state_dict(pretrain['model_state_dict'])
    print("Pretrained D state is loaded.")

    # Load dataset
    idx_ind = [0, 1, 3, 4, 5]
    B = 2
    dset_dict = MNIST_SUB(batch_size=B, val_batch_size=64,
                          idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    ind_tri_loader = dset_dict['train_set_ind_loader']

    # Start Adversarial Attack Test
    WLoss = Wasserstein.apply
    M = 100
    nl_Wt = 1.25
    Wt = to_raw(nl_Wt)
    ic(Wt)
    G_x = grad_asc_w_rej(ind_tri_loader, D, 3, 3,
                         Wt, WLoss, device=DEVICE)
    for img in G_x:
        ic(img.shape)
        ic(WLoss(torch.softmax(D(img), dim=-1)))
        ic(-(WLoss(torch.softmax(D(img), dim=-1)).log()))
    img = torch.cat(G_x)
    ic(img.shape)
    show_images(img)
    plt.show()
    # adv_generation(ind_tri_loader, D, B, M, nl_Wt)
