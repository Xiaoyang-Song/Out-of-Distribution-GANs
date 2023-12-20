from config import *
import scipy


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
def sink_dist_test_v2(input, target, C, device=DEVICE):
    
    all_class = torch.LongTensor([i for i in range(C)]).to(device)
    all_class_onehot = label_2_onehot(all_class, C, device)
    ##reshape into (B,N,D)
    all_class_onehot = torch.unsqueeze(all_class_onehot, -1)
    test_input = torch.unsqueeze(input, -1)
    test_batch_size = test_input.shape[0]
    test_loss_values = torch.zeros(test_batch_size, C).to(device)
    test_loss = SamplesLoss("sinkhorn", p=2, blur=1., cost = cost_matrix) #Wasserstein-1 distance
    for b in range(test_batch_size):
        input_b = test_input[b:b+1,:,:].repeat(C, 1, 1)
        test_loss_values[b] = test_loss(input_b[:,:,0], input_b, all_class_onehot[:,:,0], all_class_onehot)
    
    return test_loss_values.min(dim=1)[0]

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


def ood_wass_loss(input: torch.Tensor, device=DEVICE):
    """
    Test version of Wasserstein distances (no gradient flow)
    Args:
        input (torch.Tensor): B x C tensor of probabilities
        C (int): number of classes
        device (_type_, optional): device. Defaults to DEVICE.

    Returns: torch.Tensor: A 1D tensor of Wasserstein distances
    """
    # TODO: Add assertion check
    assert len(input.shape) == 2, 'Expect input tensor to have shape (B, C)'
    p = input.clone()
    B, C = p.shape
    all_class = torch.LongTensor([i for i in range(C)]).to(device)
    all1hot = label_2_onehot(all_class, C, device)
    all1hot = torch.unsqueeze(all1hot, -1)
    p = torch.unsqueeze(p, -1)
    loss = torch.zeros(B, C).to(device)
    # Approximate Wasserstein distance
    WASSLOSS = SamplesLoss("sinkhorn", p=2, blur=1, cost=cost_matrix)
    for b in range(B):
        p_b = p[b:b+1, :, :].repeat(C, 1, 1)
        loss[b] = torch.tensor([WASSLOSS(p_b[c:c+1, :, 0],
                                         p_b[c:c+1:, :],
                                         all1hot[c:c+1, :, 0],
                                         all1hot[c:c+1:, :]) for c in range(C)])
    wass_dist, _ = loss.min(dim=1)
    # ic(loss)
    return wass_dist


if __name__ == "__main__":
    ic("Hello wass_loss.py")
    # TEST ood_wass_loss function
    # test_softmax = torch.rand((5, 10))
    # c = 10  # number of classes
    # wass_loss_ood = ood_wass_loss(test_softmax, c)
    # ic(wass_loss_ood.shape)

    # TEST ood_wass_loss function
    K = 5
    onehot = torch.tensor([[1, 0, 0, 0, 0]])
    uniform = torch.ones(1, 5) * 0.2
    uniform_0 = torch.tensor([[0.19, 0.21, 0.2, 0.2, 0.2]])
    example = torch.tensor([[0, 0.25, 0.25, 0.25, 0.25]])
    example_2 = torch.tensor([[0.5, 0.5]])
    print(ood_wass_loss(uniform, 'cpu'))
    print(ood_wass_loss(uniform_0, 'cpu'))
    print(ood_wass_loss(example, 'cpu'))
    print(ood_wass_loss(onehot))
    print(ood_wass_loss(example_2, 'cpu'))

    example = torch.tensor([[0.3557, 0.2983, 0.3461],
        [0.3679, 0.2906, 0.3416]])
    print(ood_wass_loss(example, 'cpu'))
