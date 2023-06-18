from models.model import MODEL_GETTER
from ood_gan import *
from models.dc_gan_model import *
from dataset import *
from config import *
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

#---------- Argument Processing  ----------#

#---------- Dataset, Path & Regime  ----------#
dset, is_within_dset, ind, ood = config['dataset'].values()
print(f"Experiment: {dset}")
root_dir, pretrained_dir, sample_dir = config['path'].values()
method, regime, observed_cls = config['experiment'].values()
print(f"Experiment regime: {regime}")
print(f"Method: {method}")
print(line())
if regime == 'Imbalanced':
    assert observed_cls is not None
    print(f"Observed Classes are: {observed_cls}")
n_ood = args.n_ood
print(f"Number of observed OoD samples (class-level): {n_ood}")
log_dir = root_dir + f"{method}/{dset}/{regime}/{n_ood}/"
ckpt_dir = root_dir + f"{method}/{dset}/{regime}/{n_ood}/"
os.makedirs(log_dir, exist_ok=True)
pretrained_dir = pretrained_dir + f"{dset}/"

#---------- Training Hyperparameters  ----------#

###---------- Image Info  ----------###
img_info, num_classes = config['dset_info'].values()
H, W, C = img_info.values()
print(f"Input Dimension: {H} x {W} x {C}")
print(f"Number of InD classes: {num_classes}")

###---------- Models  ----------###
D_model, D_config, G_model, G_config = config['model'].values()
model_getter = MODEL_GETTER(num_classes=num_classes,
                            img_info=img_info, return_DG=False)

###---------- Trainer  ----------###
train_config = config['train_config']
mc_num = train_config['mc']
max_epoch = train_config['max_epochs']
bsz_tri, bsz_val = train_config['bsz_tri'], train_config['bsz_val']
beta = train_config['beta']
ood_bsz = train_config['ood_bsz']

###---------- Optimizer  ----------###
lr, beta1, beta2 = train_config['optimizer'].values()
print(f"Hyperparameters: beta={beta} & lr={lr} & B_InD: {bsz_tri} & B_OoD: {ood_bsz}")

#---------- Evaluation Configuration  ----------#
eval_config = config['eval_config'].values()
each_cls, cls_idx = eval_config
if each_cls:
    assert cls_idx is not None
print("Finished Processing Input Arguments.")

#---------- Experiments Starts Here  ----------#
start = time.time()

#---------- GPU information  ----------#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

#---------- Dataset & Evaler  ----------#
# note that ind and ood are deprecated for non-mnist experiment
dset = DSET(dset, is_within_dset, bsz_tri, bsz_val, ind, ood)
evaler = EVALER(dset.ind_train, dset.ind_val, dset.ind_val_loader,
                dset.ood_val, dset.ood_val_loader,
                n_ood
            , log_dir, method, num_classes)
# Load OoD
fname = sample_dir + f"{dset.name}/OOD-{regime}-{n_ood}.pt"
ood_img_batch, ood_img_label = torch.load(fname)
# assert len(ood_img_label) == n_ood * 2

for mc in range(mc_num):
    mc_start = time.time()
    print(f"Monte Carlo Iteration {mc}")
    ##### logging information #####
    writer_name = log_dir + f"[{dset.name}]-[{n_ood}]-[{regime}]-[{mc}]"
    ckpt_name = f'[{dset.name}]-[{n_ood}]-[{regime}]-[{mc}]'
    print(D_model)
    print(G_model)
    model = model_getter(D_model, D_config, G_model, G_config)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2))
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=100, gamma=0.1)  # no scheduler

    # Training dataset
    ind_tri_loader = dset.ind_train_loader
    ind_val_loader = dset.ind_val_loader
    # Trainer
    criterion = nn.CrossEntropyLoss()
    # Simple training loop
    iter_count_train = 0
    iter_count_val = 0
    for epoch in tqdm(range(max_epoch)):
        # Training
        model.train()
        train_loss, train_acc, wass = [], [], []
        for idx, (img, labels) in enumerate(ind_tri_loader):
            img = img.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(img)
            # Sample 10 ood image from the seen OoD set
            ood_idx = np.random.choice(
                len(ood_img_batch), min(len(ood_img_batch), ood_bsz), replace=False)
            ood_img = ood_img_batch[ood_idx, :, :, :].to(DEVICE)
            ood_logits = model(ood_img)
            # ic(ood_logits.shape)
            wass_loss = batch_wasserstein(ood_logits)
            loss = criterion(logits, labels) - beta * wass_loss
            loss.backward()
            # ic(loss)
            optimizer.step()
            # Append training statistics
            acc = (torch.argmax(logits, dim=1) ==
                   labels).sum().item() / labels.shape[0]
            train_acc.append(acc)
            train_loss.append(loss.detach().item())
            wass.append(wass_loss.detach().item())
            # print(f"Step {iter_count_train} | training acc: {acc} | wass: {-torch.log(-wass_loss)} | loss: {loss}")
            # pretrain_writer.add_scalar("Training/Accuracy", acc, iter_count_train)
            # pretrain_writer.add_scalar("Training/Loss", loss.detach().item(), iter_count_train)
            iter_count_train += 1

        # pretrain_writer.add_scalar("Training/Accuracy (Epoch)", np.mean(train_acc), epoch)
        # pretrain_writer.add_scalar("Training/Loss (Epoch)", np.mean(train_loss), epoch)
        print(f"\nEpoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
                | training acc: {np.mean(train_acc)} | Wass Loss {np.mean(wass)}")
        # Evaluation
        # scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for idx, (img, labels) in enumerate(ind_val_loader):
                img, labels = img.to(DEVICE), labels.to(DEVICE)
                logits = model(img)
                loss = criterion(logits, labels)
                acc = (torch.argmax(logits, dim=1) ==
                       labels).sum().item() / labels.shape[0]
                val_acc.append(acc)
                val_loss.append(loss.detach().item())
                iter_count_val += 1

            print(f"Epoch  # {epoch + 1} | validation loss: {np.mean(val_loss)} \
                | validation acc: {np.mean(val_acc)}")
    with torch.no_grad():
        # Evaluation
        torch.save(model.state_dict(),
                   log_dir + f"model-[{n_ood}]-[{max_epoch}]-[{mc}].pt")
        print("Model Checkpoint Saved!")
        evaler.evaluate(model, f'mc={mc}', None,  False, ood)
        test_backbone_D(model, dset.ind_val_loader)
        mc_stop = time.time()
        print(f"MC #{mc} time spent: {np.round(mc_stop - mc_start, 2)}s | About {np.round((mc_stop-mc_start)/60, 1)} mins")

# Display & save statistics
evaler.display_stats()
torch.save(evaler, log_dir + "eval.pt")
print("EVALER & Stats saved successfully!")

stop = time.time()
print(f"Training time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")
