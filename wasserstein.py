from wass_loss import *
from torch.autograd import Function
from dataset import *
from utils import *


def batch_wasserstein(x):
    # Input to this function is a batch of logits
    WLoss = Wasserstein.apply

    # Traditional way
    # score = WLoss(torch.softmax(x, dim=-1))
    # print(score)
    # print(torch.softmax(x, dim=-1))
    score = 1 - torch.max(torch.softmax(x, dim=-1), dim=-1)[0]
    # print(score)
    # return WLoss(torch.softmax(x, dim=-1))
    # Shortcut way
    return torch.mean(score)

def batch_dynamic_wasserstein(x):
    # Input to this function is a batch of logits
    WLoss = Dynamic_Wasserstein.apply
    # print(WLoss(torch.softmax(x, dim=-1)))
    # print(torch.softmax(x, dim=-1))
    return WLoss(torch.softmax(x, dim=-1))


def single_wasserstein(x):
    WLoss = Wasserstein.apply
    return WLoss(torch.softmax(x, dim=-1))


class Dynamic_Wasserstein(Function):
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

        all_class = torch.LongTensor([i for i in range(1)]).to(device)
        all1hot = label_2_onehot(all_class, C, device)
        all1hot = torch.unsqueeze(all1hot, -1)
        WASSLOSS = SamplesLoss("sinkhorn", p=2, blur=1.)
        p = torch.unsqueeze(p, -1)

        all1hot = all1hot.repeat(B,1,1)
        loss = WASSLOSS(p[:,:,0], p, all1hot[:,:,0], all1hot).mean()
        ctx.save_for_backward(all1hot, p)

        return loss

    @staticmethod
    def backward(ctx, upstream_grad):
        all1hot, p = ctx.saved_tensors
        B, C, _ = p.shape

        OOD_loss = SamplesLoss("sinkhorn", p=2, blur=1., potentials=True)
        OOD_f, OOD_g = OOD_loss(p[:,:,0], p, all1hot[0:1].repeat(B,1,1)[:,:,0], all1hot[0:1].repeat(B,1,1))

        grad = upstream_grad * OOD_f
    
        return grad

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
        WASSLOSS = SamplesLoss("sinkhorn", p=2, blur=0.05, cost=cost_matrix)
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
        ic(values)
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
    k = torch.tensor([[1.0, 2.0], [3,1]], requires_grad=True)
    a = -batch_wasserstein(k)
    print("a", a)
    a = -batch_dynamic_wasserstein(k)
    print("a", a)
    a.backward()
    print(k.grad.data)
    b = batch_dynamic_wasserstein(torch.tensor([[1.0-0.1366, 2.0+0.1366]], requires_grad=True))
    print(b)
    b = batch_wasserstein(torch.tensor([[1.0-0.1366, 2.0+0.1366]], requires_grad=True))
    print(b)
    a = torch.tensor([[3.0, 6.0]], requires_grad=True)
    c = torch.sum(a)
    c.backward()
    print(a.grad.data)
