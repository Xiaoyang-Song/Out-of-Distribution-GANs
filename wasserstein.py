from wass_loss import *
from torch.autograd import Function
from dataset import *
from utils import *


def batch_wasserstein(x):
    # Input to this function is a batch of logits
    WLoss = Wasserstein.apply
    # print(WLoss(torch.softmax(x, dim=-1)))
    # print(torch.softmax(x, dim=-1))
    return WLoss(torch.softmax(x, dim=-1))


def single_wasserstein(x):
    WLoss = Wasserstein.apply
    return WLoss(torch.softmax(x, dim=-1))


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
        # print(values)
        # ic(values.mean())
        return values.mean()

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
        grad = OOD_f * upstream_grad
        # print(grad)
        return grad, None, None, None, None


if __name__ == '__main__':
    ic("Hello wasserstein.py")
    # This is not a driver class
    k = torch.tensor([[1.0, 9.0], [1.0, 1.0]], requires_grad=True)
    a = batch_wasserstein(k)
    print(a)
    a.backward()
    print(k.grad.data)
    b = batch_wasserstein(torch.tensor([[1.0-0.5285, 2.0+0.1664]], requires_grad=True))
    print(b)
    a = torch.tensor([[3.0, 6.0]], requires_grad=True)
    c = torch.sum(a)
    c.backward()
    print(a.grad.data)
