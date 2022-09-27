from config import *


def cost_matrix(X, Y):
    # TODO: Change this to more generic version
    if len(X.shape) == 2:
        N, D = X.shape
        M, D = Y.shape
        return (1 - torch.eye(N, M)).to(DEVICE)
    elif len(X.shape) == 3:
        B, N, D = X.shape
        B, M, D = Y.shape
        return torch.unsqueeze(1 - torch.eye(N, M), 0).repeat(B, 1, 1).to(DEVICE)
    else:
        assert False, 'Unexpected dimension of X or Y. Expect (B, N, D) or (N, D)'


def label_2_onehot(label, C, device):
    # transform the InD labels into one-hot vector
    assert type(label) == torch.Tensor

    size = label.shape[0]
    if len(label.shape) == 1:
        label = torch.unsqueeze(label, 1)

    label = label % C

    label_onehot = torch.FloatTensor(size, C).to(device)

    label_onehot.zero_()
    label_onehot.scatter_(1, label, 1)
    return label_onehot

# The two functinos below are borrowed and MODIFIED from the WOOD LOSS PAPER:
# Names of the original functions
# sink_dist_test -> ind_wass_loss
# sink_dist_test_v2 -> ood_wass_loss


def ind_wass_loss(input: torch.Tensor, target: torch.Tensor, C: int, device=DEVICE):
    # TODO: Add assertion check
    # TODO: Add docstrings for illustration
    test_label_onehot = label_2_onehot(target, C, device)
    test_label_onehot = torch.unsqueeze(test_label_onehot, -1)
    test_input = torch.unsqueeze(input, -1)
    # Loss value for InD samples
    # Wasserstein-1 distance
    # test_loss = SamplesLoss("sinkhorn", p=2, blur=1., cost=cost_matrix)
    test_loss = SamplesLoss("sinkhorn", p=2, blur=1.)
    test_loss_value = test_loss(
        test_input[:, :, 0], test_input, test_label_onehot[:, :, 0], test_label_onehot)

    return test_loss_value


def ood_wass_loss(input: torch.Tensor, C: int, device=DEVICE):
    # TODO: Add assertion check
    # ic(input.requires_grad)
    all_class = torch.LongTensor([i for i in range(C)]).to(device)
    all_class_onehot = label_2_onehot(all_class, C, device)
    # reshape into (B,N,D)
    all_class_onehot = torch.unsqueeze(all_class_onehot, -1)
    test_input = torch.unsqueeze(input, -1)
    test_batch_size = test_input.shape[0]
    test_loss_values = torch.zeros(test_batch_size, C).to(device)
    # Approximate Wasserstein distance
    test_loss = SamplesLoss("sinkhorn", p=2, blur=1., cost=cost_matrix)
    for b in range(test_batch_size):
        input_b = test_input[b:b+1, :, :].repeat(C, 1, 1)
        # Modified the line below
        test_loss_values[b] = torch.tensor([test_loss(input_b[c:c+1, :, 0],
                                            input_b[c:c+1:, :],
                                            all_class_onehot[c:c+1, :, 0],
                                            all_class_onehot[c:c+1:, :]) for c in range(C)])
    ans = test_loss_values.min(dim=1)[0]
    ans.requires_grad = True
    return ans


if __name__ == "__main__":
    ic("Hello wass_loss.py")
    # TEST ood_wass_loss function
    # test_softmax = torch.rand((5, 10))
    # c = 10  # number of classes
    # wass_loss_ood = ood_wass_loss(test_softmax, c)
    # ic(wass_loss_ood.shape)

    # TEST ood_wass_loss function 2
    K = 5
    c1 = torch.tensor([[0.01, 0, 0.99, 0, 0]], requires_grad=True)
    c1_5 = torch.tensor([[0.01, 0, 0.8, 0.19, 0]])
    # Examples
    c0 = torch.tensor([[0.01, 0, 0.7, 0.19, 0.1]])
    c0 = torch.tensor([[0.10, 0.1, 0.4, 0.30, 0.1]])
    c2 = torch.ones((5)) * 0.2
    # ic(ood_wass_loss(c1, 5).requires_grad)
    # ic(ood_wass_loss(c1_5, 5))
    # ic(ood_wass_loss(c2.unsqueeze(0), 5))

    def wass(x, K):
        return -torch.log(ood_wass_loss(x, K))
    ic(wass(c1, K))
    ic(wass(c1_5, K))
    ic(wass(c0, K))
    ic(-torch.log(ood_wass_loss(c2.unsqueeze(0), 5)))
