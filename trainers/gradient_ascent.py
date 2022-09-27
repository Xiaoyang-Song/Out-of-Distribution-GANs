from config import *
from wass_loss import *
from datasets import *
from utils import *
from ood_gan_trainer import *
from models.dc_gan_model import dc_discriminator, dc_generator


def grad_asc_w_rej(ind_loader, D, B, M, Wt, device=DEVICE):
    D.requires_grad = False
    G_x = []
    for i, (x, y) in enumerate(ind_loader):
        x_g = x.clone().requires_grad_().to(device)
        g_x = [ad_atk(D, img, Wt) for img in x_g]
        G_x.append(g_x)
        if i >= M:
            return G_x


def ad_atk(D, x, Wt):
    assert x.requires_grad
    assert not D.requires_grad
    lr = 1
    i = 1
    while True:
        # Aim to minimize W
        W = -zero_softmax_loss(D(x))
        if W < Wt:
            print(f"Adversarial Attack done in {i} iterations")
            return x.detach()
        W.backward()
        grad = x.grad.data
        ic(grad.shape)
        # Update
        dx = lr * grad / grad.norm()  # Normalize
        x.data -= lr * dx  # Can still do Gradient Dscent
        x.grad.zero_()
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

    # Load dataset
    idx_ind = [0, 1, 3, 4, 5]
    dset_dict = MNIST_SUB(batch_size=128, val_batch_size=64,
                          idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    ind_tri_loader = dset_dict['train_set_ind_loader']
    # Start Gradient Ascent
    Gx = grad_asc_w_rej(ind_tri_loader, D, 2, 1, 1.5)
