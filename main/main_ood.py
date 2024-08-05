from ood_gan import *
from models.dc_gan_model import *
from dataset import *
from config import *
from models.model import *
from eval import *
import argparse
import time
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Training configuration file')
parser.add_argument('--n_ood', help="Number of OoD samples", type=int)
args = parser.parse_args()
assert args.config is not None, 'Please specify the config .yml file to proceed.'
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

########## Argument Processing  ##########
#---------- Dataset, Path & Regime  ----------#
dset, is_within_dset, ind, ood = config['dataset'].values()
print(f"Experiment: {dset}")
root_dir, pretrained_dir, sample_dir = config['path'].values()
method, regime, observed_cls = config['experiment'].values()
print(f"Experiment regime: {regime}")
print(f"Method: {method}")
print(line())

n_ood = args.n_ood
print(f"Number of observed OoD samples (class-level): {n_ood}")
log_dir = root_dir + f"{method}/{dset}/{regime}/{n_ood}/"
ckpt_dir = root_dir + f"{method}/{dset}/{regime}/{n_ood}/"
os.makedirs(log_dir, exist_ok=True)


#---------- Training Hyperparameters  ----------#

###---------- Image Info  ----------###
img_info, num_classes = config['dset_info'].values()
H, W, C = img_info.values()
print(f"Input Dimension: {H} x {W} x {C}")
print(f"Number of InD classes: {num_classes}")

###---------- Models  ----------###
D_model, D_config, G_model, G_config = config['model'].values()
model_getter = MODEL_GETTER(num_classes=num_classes, img_info=img_info, return_DG=True)

###---------- Trainer  ----------###
train_config = config['train_config']
mc_num = train_config['mc']
max_epoch = train_config['max_epochs']
bsz_tri, bsz_val = train_config['bsz_tri'], train_config['bsz_val']
ood_bsz = train_config['ood_bsz']

w_ce, w_loss, w_dist = train_config['hp'].values()
hp = HParam(ce=w_ce, wass=w_loss, dist=w_dist)
scaling = train_config['scaling']
score = train_config['score']
T = train_config['T']
d_step_ratio = train_config['d_step_ratio']
g_step_ratio = train_config['g_step_ratio']
noise_dim = train_config['noise_dim']
n_steps_log = train_config['logging']['n_steps_log']
n_epochs_save = train_config['logging']['n_epochs_save']
print(f"Number of Epochs: {max_epoch}.")

###---------- Optimizer  ----------###
d_lr, g_lr, beta1, beta2 = train_config['optimizer'].values()
print(f"Hyperparameters: lambda_ce={w_ce} & lambda_w={w_loss} & scaling={scaling} & d_lr={d_lr} & g_lr={g_lr} & B_InD: {bsz_tri} & B_OoD: {ood_bsz} & n_d: {d_step_ratio} & n_g: {g_step_ratio}")
print(f"Score Function: {score}")
#---------- Evaluation Configuration  ----------#
eval_config = config['eval_config'].values()
each_cls, cls_idx, n_lr = eval_config
if each_cls:
    assert cls_idx is not None
print("Finished Processing Input Arguments.")

########## Experiment Starts Here  ##########
start = time.time()
# ic("HELLO GL!")

#---------- GPU information  ----------#
if torch.cuda.is_available():
    print(f"-- Current Device: {torch.cuda.get_device_name(0)}")
    print(f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("-- Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("-- Unfortunately, we are only using CPUs now.")
#---------- Dataset & Evaler  ----------#
# note that ind and ood are deprecated for non-mnist experiment
dset = DSET(dset, is_within_dset, bsz_tri, bsz_val, ind, ood)
evaler = EVALER(dset.ind_train, dset.ind_val, dset.ind_val_loader,
                dset.ood_val, dset.ood_val_loader,
                n_ood, log_dir, method, num_classes, n_lr)
# Load OoD
# fname = sample_dir + f"{dset.name}/OOD-{regime}-{n_ood}.pt"
fname = sample_dir + f"{dset.name}/OOD-{regime}-{n_ood}.pt"
ood_img_batch, ood_img_label = torch.load(fname)
print(ood_img_label)
print(Counter(np.array(ood_img_label)))

#---------- Monte Carlo Simulation  ----------#
for mc in range(mc_num):
    mc_start = time.time()
    print(f"Monte Carlo Iteration {mc}")

    ###---------- logging information  ----------###
    writer_name = log_dir + f"[{dset.name}]-[{n_ood}]-[{regime}]-[{mc}]"
    ckpt_name = f'[{dset.name}]-[{n_ood}]-[{regime}]-[{mc}]'

    ###---------- models  ----------###
    print(D_model)
    print(G_model)
    D, G = model_getter(D_model, D_config, G_model, G_config)
    # evaler.evaluate(D, f'mc={mc}', G, each_cls, cls_idx)
    # ic("Test Done.")

    ###---------- checkpoint loading (if necessary)  ----------###
    # TODO: optimize the implementation later.
    # ckpt = "/scratch/sunwbgt_root/sunwbgt98/xysong/energy_ood/CIFAR/snapshots/pretrained/[CIFAR100-SVHN]-pretrained-classifier.pt"
    # D.load_state_dict(torch.load(ckpt))
    # name = f"[{dset.name}]-[{n_ood}]-[{regime}]-[0]_[9]"
    # ckpt = f"/scratch/sunwbgt_root/sunwbgt98/xysong/Out-of-Distribution-GANs/checkpoint/OOD-GAN/{dset.name}/{regime}/{n_ood}/{name}.pt"

    # D.load_state_dict(torch.load(ckpt)['D-state'])
    # G.load_state_dict(torch.load(ckpt)['G-state'])
    # print('Successfully Load checkpoints')
    ###---------- optimizers  ----------###
    D_solver = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(beta1, beta2))
    G_solver = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(beta1, beta2))
    # D_solver = torch.optim.SGD(D.parameters(), lr=d_lr, momentum=0.9)
    # G_solver = torch.optim.SGD(G.parameters(), lr=g_lr, momentum=0.9)

    # D_scheduler = torch.optim.lr_scheduler.StepLR(D_solver, 10, gamma=0.1)
    # G_scheduler = torch.optim.lr_scheduler.StepLR(G_solver, 10, gamma=0.1)

    ###---------- dataset  ----------###
    ind_loader = dset.ind_train_loader
    ###---------- trainer  ----------###
    trainer = OOD_GAN_TRAINER(D=D, G=G,
                              noise_dim=noise_dim,
                              num_classes=num_classes,
                              bsz_tri=bsz_tri,
                              d_steps_ratio=d_step_ratio,
                              g_steps_ratio=g_step_ratio,
                              hp=hp,
                              scaling=scaling,
                              score=score,
                              T=T,
                              max_epochs=max_epoch,
                              ood_bsz=ood_bsz,
                              writer_name=writer_name,
                              ckpt_name=ckpt_name,
                              ckpt_dir=ckpt_dir,
                              n_steps_log=n_steps_log,
                              n_epochs_save=n_epochs_save,
                              ind_val_loader = dset.ind_val_loader,
                              ood_val_loader = dset.ood_val_loader)

    # Used for complex dataset
    trainer.train(ind_loader, ood_img_batch, D_solver,
                  G_solver, pretrainedD=None, checkpoint=None)

    ###---------- evaluation  ----------###
    evaler.evaluate(D, f'mc={mc}', G, each_cls, cls_idx)
    test_backbone_D(D, dset.ind_val_loader)
    mc_stop = time.time()
    print(f"MC #{mc} time spent: {np.round(mc_stop - mc_start, 2)} seconds | About {np.round((mc_stop-mc_start)/60, 2)} minutes | About {np.round((mc_stop-mc_start)/3600, 2)} hours")

# Display & save statistics
evaler.display_stats()
evaler.save(log_dir + "eval.pt")
print("EVALER & Stats saved successfully!")

stop = time.time()
print(f"Training time: {np.round(stop - start, 2)} seconds | About {np.round((stop-start)/60, 2)} minutes | About {np.round((stop-start)/3600, 2)} hours")
