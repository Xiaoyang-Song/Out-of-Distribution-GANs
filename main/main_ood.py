from ood_gan import *
from models.dc_gan_model import *
from dataset import *
from config import *
from eval import *
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--mc', help='Number of MC', type=int)
parser.add_argument('--num_epochs', help='Number of Epochs', type=int)
parser.add_argument('--balanced', help='Balanced', type=str)
parser.add_argument('--n_ood', help='Number of observed OoD', type=int)
args = parser.parse_args()
# ic(args.balanced)
# assert False
start = time.time()
ic("HELLO GL!")
ic(torch.cuda.is_available())
if torch.cuda.is_available():
    ic(torch.cuda.get_device_name(0))
##### Config #####
ood_bsz = args.n_ood
log_dir = f"../checkpoint/MNIST/Test/{ood_bsz}/"
ckpt_dir = f"../checkpoint/MNIST/Test/{ood_bsz}/"
pretrained_dir = f"../checkpoint/pretrained/mnist/"
##### Hyperparameters #####
hp = HParam(ce=1, wass=0.1, dist=1)
noise_dim = 96
img_info = {'H': 28, 'W': 28, 'C': 1}
max_epoch = args.num_epochs
##### Dataset #####
dset = DSET('mnist', 50, 128, [2, 3, 6, 8, 9], [1, 7])
evaler = EVALER(dset.ind_train, dset.ind_val, dset.ood_val, ood_bsz, log_dir)

##### Monte Carlo config #####
MC_NUM = args.mc

for mc in range(MC_NUM):
    mc_start = time.time()
    ic(f"Monte Carlo Iteration {mc}")
    ##### logging information #####
    writer_name = log_dir + f"MNIST-[{ood_bsz}]-[{mc}]"
    ckpt_name = f'MNIST-[{ood_bsz}]-balanced-[{mc}]'

    D = DC_D(5, img_info).to(DEVICE)
    ckpt = torch.load(pretrained_dir + "mnist-[23689]-D.pt")
    D.load_state_dict(ckpt['model_state_dict'])
    G = DC_G().to(DEVICE)
    # ckpt = torch.load(pretrained_dir + "mnist-[23689]-G.pt")
    # G.load_state_dict(ckpt['model_state_dict'])
    D_solver = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    G_solver = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # Training dataset
    ind_loader = dset.ind_train_loader
    regime = args.balanced
    ood_img_batch, ood_img_label = dset.ood_sample(ood_bsz, regime, None)
    ic(ood_img_label)

    torch.save((ood_img_batch, ood_img_label),
               log_dir + f"x_ood-[{ood_bsz}]-[{mc}]")

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
    # Evaluation
    evaler.compute_stats(D, f'mc={mc}', G, True, [1, 7])
    mc_stop = time.time()
    ic(f"MC #{mc} time spent: {np.round(mc_stop - mc_start, 2)}s | About {np.round((mc_stop-mc_start)/60, 1)} mins")

# Display & save statistics
evaler.display_stats()
torch.save(evaler, log_dir + "eval.pt")
ic("EVALER & Stats saved successfully!")

stop = time.time()
ic(f"Training time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")
