from ood_gan import *


ckpt_dir = "../checkpoint/log/"
max_epochs = 2

##### Hyperparameters #####
hp = HParam(ce=1, wass=0.1, dist=1)
noise_dim = 96
img_info = {'H':28,'W':28,'C':1}

##### logging information #####
