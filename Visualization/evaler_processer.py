from eval import EVALER
import torch

# evaler = torch.load("checkpoint/OOD-GAN/FashionMNIST/Balanced/128/eval.pt")
evaler = torch.load("res/eval.pt", map_location=torch.device('cpu'))
print(evaler.n_ood)
