from ood_gan import *
from models.dc_gan_model import *
from dataset import *
from config import *

print("HELLO GL!")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

log_dir = "../checkpoint/log/"
ckpt_dir = "../checkpoint/"
pretrained_dir = "../checkpoint/pretrained/mnist/"

##### Config #####
ood_bsz = 32

##### Hyperparameters #####
hp = HParam(ce=1, wass=0.1, dist=1)
noise_dim = 96
img_info = {'H': 28, 'W': 28, 'C': 1}
max_epoch = 2
##### logging information #####
writer_name = log_dir + f"MNIST-[{ood_bsz}]"
ckpt_name = f'MNIST-[{ood_bsz}]-balanced'
##### Dataset #####
dset = DSET('mnist', 50, 128, [2, 3, 6, 8, 9], [1, 7])

D = DC_D(5, img_info).to(DEVICE)
ckpt = torch.load(pretrained_dir + "mnist-[23689]-D.pt")
D.load_state_dict(ckpt['model_state_dict'])
G = DC_G().to(DEVICE)
ckpt = torch.load(pretrained_dir + "mnist-[23689]-G.pt")
G.load_state_dict(ckpt['model_state_dict'])
D_solver = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
G_solver = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
# Training dataset
ind_loader = dset.ind_train_loader
ood_img_batch, ood_img_label = dset.get_ood_equal(ood_bsz)
ic(ood_img_label)

# Trainer
trainer = OOD_GAN_TRAINER(D=D, G=G,
                          noise_dim=noise_dim,
                          bsz_tri=50,
                          gd_steps_ratio=1,
                          hp=hp,
                          max_epochs=max_epoch,
                          writer_name=writer_name,
                          ckpt_name=ckpt_name,
                          ckpt_dir=ckpt_dir,
                          n_steps_log=5)
trainer.train(ind_loader, ood_img_batch, D_solver, G_solver,
              D.encoder, pretrainedD=None, checkpoint=None)
