from config import *
from wass_loss import *
from datasets import *
from utils import *
from ood_gan_trainer import *
from models.dc_gan_model import dc_discriminator, dc_generator
from wasserstein import Wasserstein


def grad_asc_w_rej(ind_loader, D, B, M, Wt, WLoss, device=DEVICE):
    # D.requires_grad = False
    G_x = []
    for i, (x, y) in enumerate(ind_loader):

        for img in x:
            p = img.clone().unsqueeze(0).to(device)
            p.requires_grad_()
            # ic(p.shape)
            g_x = ad_atk(D, p, Wt, WLoss)
            G_x.append(g_x)

        if i >= M:
            return G_x


def ad_atk(D, x, Wt, WLoss):
    assert x.requires_grad
    ic(x.shape)
    x = nn.Parameter(x)
    optimizer = torch.optim.Adam(
        [{"params": x}], lr=1e-3)

    lr = 1
    i = 1
    while True:
        # Aim to minimize W
        optimizer.zero_grad()
        logit = D(x)
        # ic(logit.shape)
        # W = neg_zero_softmax_loss(logit)
        W = WLoss(torch.softmax(logit, dim=-1))
        ic(W)
        if W < Wt:
            print(f"Adversarial Attack done in {i} iterations")
            return x.detach()
        # x.retain_grad()
        W.backward()
        optimizer.step()
        # grad = x.grad.data
        # ic(grad.shape)
        # # Update
        # dx = lr * grad / grad.norm()  # Normalize
        # x.data -= lr * dx  # Can still do Gradient Dscent
        # x.grad.zero_()
        i += 1


def find_W_t():
    return 1.5


if __name__ == '__main__':
    ic("Hello gradient_ascent.py")

    # Load pretrained_D
    img_info = {'H': 28, 'W': 28, 'C': 1}
    D = dc_discriminator(img_info, GAN_TYPE.OOD).to(DEVICE)
    path = 'checkpoint/pretrained_D_low.pt'
    pretrain = torch.load(path)
    D.load_state_dict(pretrain['model_state_dict'])
    print("Pretrained D state is loaded.")
    # D.parameters().retain_grad()
    # ic(D[0].weight.is_leaf)

    # Load dataset
    idx_ind = [0, 1, 3, 4, 5]
    dset_dict = MNIST_SUB(batch_size=2, val_batch_size=64,
                          idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    ind_tri_loader = dset_dict['train_set_ind_loader']
    # batch = next(iter(ind_tri_loader))[0]
    # Start Gradient Ascent
    WLoss = Wasserstein.apply
    G_x = grad_asc_w_rej(ind_tri_loader, D, 3, 1, -0.25, WLoss, device=DEVICE)
    for img in G_x:
        ic(img.shape)
        ic(WLoss(torch.softmax(D(img), dim=-1)))
        ic((-WLoss(torch.softmax(D(img), dim=-1))).log())
    img = torch.cat(G_x)
    ic(img.shape)
    show_images(img)
    plt.show()
