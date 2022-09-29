from wass_loss import *
from torch.autograd import Function
from models.dc_gan_model import dc_discriminator, dc_generator
from models.gans import *
from dataset import *
from utils import *


class Wasserstein(Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, device=DEVICE):
        """
        Forward pass for Wasserstein distance metric computation

        Args:
            ctx (_type_): static object
            p (torch.tensor): B x C predicted probability vector
            device (_type_, optional): default to DEVICE
        """
        p = input.clone()
        B, C = input.shape
        # ic(B)
        # ic(C)
        # ic(p.requires_grad_)
        all_class = torch.LongTensor([i for i in range(C)]).to(device)
        all1hot = label_2_onehot(all_class, C, device)
        all1hot = torch.unsqueeze(all1hot, -1)
        # 2-Wasserstein distance with binary cost matrix
        WASSLOSS = SamplesLoss("sinkhorn", p=2, blur=1., cost=cost_matrix)
        p = torch.unsqueeze(p, -1)
        # Compute Wasserstein distance
        loss = torch.zeros(B, C).to(device)
        for b in range(B):
            p_b = p[b:b+1, :, :].repeat(C, 1, 1)
            # ic(p_b.shape)
            # ic(p_b[:, :, 0].shape)
            # ic(all1hot.shape)
            # loss[b] = WASSLOSS(p_b[:, :, 0], p_b, all1hot[:, :, 0], all1hot)
            loss[b] = torch.tensor([WASSLOSS(p_b[c:c+1, :, 0],
                                             p_b[c:c+1:, :],
                                             all1hot[c:c+1, :, 0],
                                             all1hot[c:c+1:, :]) for c in range(C)])
        values, idx = torch.min(loss, dim=1)

        # Save for backward pass
        ctx.save_for_backward(all1hot, p, idx)
        return -values.mean()

    @staticmethod
    def backward(ctx, upstream_grad):
        all1hot, p, idx = ctx.saved_tensors
        B, C, _ = p.shape
        WASSLOSSDUAL = SamplesLoss(
            "sinkhorn", p=2, blur=1., potentials=True, cost=cost_matrix)

        OOD_f = torch.zeros(B, C).to(DEVICE)
        for b in range(B):
            p_b = p[b:b+1, :, :].repeat(C, 1, 1)
            # f, _ = WASSLOSSDUAL(p_b[:, :, 0], p_b, all1hot[:, :, 0], all1hot)
            f = []
            for c in range(C):
                d, _ = WASSLOSSDUAL(p_b[c:c+1, :, 0],
                                    p_b[c:c+1:, :],
                                    all1hot[c:c+1, :, 0],
                                    all1hot[c:c+1:, :])
                f.append(d)
            f = torch.cat(f)
            OOD_f[b] = f[idx[b]]

        grad = torch.zeros([B, C]).to(DEVICE)
        grad = -OOD_f
        return grad, None, None, None, None


if __name__ == '__main__':
    ic("Hello wasserstein.py")

    img_info = {'H': 28, 'W': 28, 'C': 1}
    D = dc_discriminator(img_info, GAN_TYPE.OOD).to(DEVICE)
    path = 'checkpoint/pretrained_D_low.pt'
    pretrain = torch.load(path)
    D.load_state_dict(pretrain['model_state_dict'])
    print("Pretrained D state is loaded.")

    c0 = torch.tensor(
        [[0.01, 0, 0.8, 0.19, 0]], requires_grad=True)
    WLoss = Wasserstein.apply
    # W = WLoss(c0)
    # ic(W.requires_grad_)
    # W.backward()
    # ic(c0.grad.data.shape)

    # Load dataset
    idx_ind = [0, 1, 3, 4, 5]
    dset_dict = MNIST_SUB(batch_size=128, val_batch_size=64,
                          idx_ind=idx_ind, idx_ood=[2], shuffle=True)
    ind_tri_loader = dset_dict['train_set_ind_loader']
    batch = next(iter(ind_tri_loader))[0]
    # Start Gradient Ascent
    # Gx = grad_asc_w_rej(ind_tri_loader, D, 2, 1, 1.5)
    batch.requires_grad_()
    ic(batch.shape)
    ic(D(batch).shape)
    ic(batch.requires_grad_)
    logit = D(batch)
    ic(logit.requires_grad_)
    W = -WLoss(torch.softmax(logit, dim=-1)).log()
    ic(W.requires_grad_)
    batch.retain_grad()
    W.backward()

    ic(batch.grad.data.shape)
    # Test backward pass
