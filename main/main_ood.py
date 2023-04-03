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
args = parser.parse_args()
assert args.config is not None, 'Please specify the config .yml file to proceed.'
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

########## Argument Processing  ##########
#---------- Dataset, Path & Regime  ----------#
dset, is_within_dset, ind, ood = config['dataset'].values()
ic(f"Experiment: {dset}")
root_dir, pretrained_dir = config['path'].values()
method, regime, observed_cls = config['experiment'].values()
ic(f"Experiment regime: {regime}")
ic(f"Method: {method}")
print(line())
if regime == 'Imbalanced':
    assert observed_cls is not None
    ic(f"Observed Classes are: {observed_cls}")
ood_bsz = config['n_ood']
ic(f"Number of observed OoD samples (class-level): {ood_bsz}")
log_dir = root_dir + f"{method}/{dset}/{regime}/{ood_bsz}/"
ckpt_dir = root_dir + f"{method}/{dset}/{regime}/{ood_bsz}/"
os.makedirs(log_dir, exist_ok=True)
pretrained_dir = pretrained_dir + f"{dset}/"
#---------- Training Hyperparameters  ----------#
###---------- Image Info  ----------###
img_info, num_classes = config['dset_info'].values()
H, W, C = img_info.values()
ic(f"Input Dimension: {H} x {W} x {C}")
ic(f"Number of InD classes: {num_classes}")
###---------- Models  ----------###
D_model, D_config, G_model, G_config = config['model'].values()
model_getter = MODEL_GETTER(num_classes=num_classes,
                            img_info=img_info, return_DG=True)
###---------- Trainer  ----------###
train_config = config['train_config']
mc_num = train_config['mc']
max_epoch = train_config['max_epochs']
bsz_tri, bsz_val = train_config['bsz_tri'], train_config['bsz_val']
w_ce, w_loss, w_dist = train_config['hp'].values()
hp = HParam(ce=w_ce, wass=w_loss, dist=w_dist)
gd_step_ratio = train_config['gd_step_ratio']
noise_dim = train_config['noise_dim']
n_steps_log = train_config['logging']['n_steps_log']
###---------- Optimizer  ----------###
lr, beta1, beta2 = train_config['optimizer'].values()
#---------- Evaluation Configuration  ----------#
eval_config = config['eval_config'].values()
each_cls, cls_idx, n_lr = eval_config
if each_cls:
    assert cls_idx is not None
ic("Finished Processing Input Arguments.")
########## Experiment Starts Here  ##########
start = time.time()
ic("HELLO GL!")
#---------- GPU information  ----------#
ic(torch.cuda.is_available())
if torch.cuda.is_available():
    ic(torch.cuda.get_device_name(0))
#---------- Dataset & Evaler  ----------#
# note that ind and ood are deprecated for non-mnist experiment
dset = DSET(dset, is_within_dset, bsz_tri, bsz_val, ind, ood)
evaler = EVALER(dset.ind_train, dset.ind_val, dset.ind_val_loader,
                dset.ood_val, dset.ood_val_loader,
                ood_bsz, log_dir, method, num_classes, n_lr)

#---------- Monte Carlo Simulation  ----------#
for mc in range(mc_num):
    mc_start = time.time()
    ic(f"Monte Carlo Iteration {mc}")
    ###---------- logging information  ----------###
    writer_name = log_dir + f"[{dset}]-[{ood_bsz}]-[{regime}]-[{mc}]"
    ckpt_name = f'[{dset}]-[{ood_bsz}]-[{regime}]-[{mc}]'
    ###---------- models  ----------###
    ic(D_model)
    ic(G_model)
    D, G = model_getter(D_model, D_config, G_model, G_config)
    # Load checkpoint if necessary
    # ckpt = torch.load(pretrained_dir + "D.pt")
    # D.load_state_dict(ckpt['model_state_dict'])
    # ic("Pretrained D states loaded!")
    # ckpt = torch.load(pretrained_dir + "G.pt")
    # G.load_state_dict(ckpt['model_state_dict'])
    ###---------- optimizers  ----------###
    D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    G_solver = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    ###---------- dataset  ----------###
    ind_loader = dset.ind_train_loader
    if regime == 'Balanced':
        ood_img_batch, ood_img_label = dset.ood_sample(ood_bsz, regime)
    elif regime == 'Imbalanced':
        ood_img_batch, ood_img_label = dset.ood_sample(
            ood_bsz, regime, observed_cls)
    ic(ood_img_label)
    torch.save((ood_img_batch, ood_img_label),
               log_dir + f"x_ood-[{ood_bsz}]-[{mc}]")

    ###---------- trainer  ----------###
    trainer = OOD_GAN_TRAINER(D=D, G=G,
                              noise_dim=noise_dim,
                              num_classes=num_classes,
                              bsz_tri=bsz_tri,
                              gd_steps_ratio=gd_step_ratio,
                              hp=hp,
                              max_epochs=max_epoch,
                              writer_name=writer_name,
                              ckpt_name=ckpt_name,
                              ckpt_dir=ckpt_dir,
                              n_steps_log=n_steps_log)
    trainer.train(ind_loader, ood_img_batch, D_solver, G_solver,
                  D.encoder, pretrainedD=None, checkpoint=None)
    ###---------- evaluation  ----------###
    evaler.evaluate(D, f'mc={mc}', G, each_cls, cls_idx)
    mc_stop = time.time()
    ic(f"MC #{mc} time spent: {np.round(mc_stop - mc_start, 2)}s | About {np.round((mc_stop-mc_start)/60, 1)} mins")

# Display & save statistics
evaler.display_stats()
torch.save(evaler, log_dir + "eval.pt")
ic("EVALER & Stats saved successfully!")

stop = time.time()
ic(f"Training time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")
