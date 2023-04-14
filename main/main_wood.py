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
                            img_info=img_info, return_DG=False)
###---------- Trainer  ----------###
train_config = config['train_config']
mc_num = train_config['mc']
max_epoch = train_config['max_epochs']
bsz_tri, bsz_val = train_config['bsz_tri'], train_config['bsz_val']
###---------- Optimizer  ----------###
lr, beta1, beta2 = train_config['optimizer'].values()
#---------- Evaluation Configuration  ----------#
eval_config = config['eval_config'].values()
each_cls, cls_idx = eval_config
if each_cls:
    assert cls_idx is not None
ic("Finished Processing Input Arguments.")
########## Experiment Starts Here  ##########
start = time.time()
ic("HELLO GL!")
#---------- GPU information  ----------#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ic(torch.cuda.is_available())
if torch.cuda.is_available():
    ic(torch.cuda.get_device_name(0))
    ic(torch.cuda.get_device_properties(0).total_memory)
    # ic(torch.cuda.getMemoryUsage(0))
    ic("Let's use", torch.cuda.device_count(), "GPUs!")

#---------- Dataset & Evaler  ----------#
# note that ind and ood are deprecated for non-mnist experiment
dset = DSET(dset, is_within_dset, bsz_tri, bsz_val, ind, ood)
evaler = EVALER(dset.ind_train, dset.ind_val, dset.ind_val_loader,
                dset.ood_val, dset.ood_val_loader,
                ood_bsz, log_dir, method, num_classes)

for mc in range(mc_num):
    mc_start = time.time()
    ic(f"Monte Carlo Iteration {mc}")
    ##### logging information #####
    writer_name = log_dir + f"[{dset}]-[{ood_bsz}]-[{regime}]-[{mc}]"
    ckpt_name = f'[{dset.name}]-[{ood_bsz}]-[{regime}]-[{mc}]'
    ic(D_model)
    ic(G_model)
    model = model_getter(D_model, D_config, G_model, G_config)
    # model = nn.DataParallel(model)
    # Load pretrained checkpoint if necessary
    # ckpt = torch.load(pretrained_dir + "mnist-[23689]-D.pt")
    # model.load_state_dict(ckpt['model_state_dict'])
    # ic('Checkpoint loaded')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)
    # Training dataset
    ind_tri_loader = dset.ind_train_loader
    ind_val_loader = dset.ind_val_loader
    if regime == 'Balanced':
        ood_img_batch, ood_img_label = dset.ood_sample(ood_bsz, regime)
    elif regime == 'Imbalanced':
        ood_img_batch, ood_img_label = dset.ood_sample(
            ood_bsz, regime, observed_cls)
    ood_img_batch = ood_img_batch.to(DEVICE)
    ic(ood_img_label)
    torch.save((ood_img_batch, ood_img_label),
               log_dir + f"x_ood-[{ood_bsz}]-[{mc}]")

    # Trainer
    criterion = nn.CrossEntropyLoss()
    # Simple training loop
    iter_count_train = 0
    iter_count_val = 0
    for epoch in tqdm(range(max_epoch)):
        # Training
        model.train()
        train_loss, train_acc, wass = [], [], []
        for idx, (img, label) in enumerate(ind_tri_loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(img)
            # Sample 10 ood image from the seen OoD set
            ood_idx = np.random.choice(
                len(ood_img_batch), min(len(ood_img_batch), 10), replace=False)
            ood_logits = model(ood_img_batch[ood_idx, :, :, :])
            wass_loss = batch_wasserstein(ood_logits)
            loss = criterion(logits, label) + 0.1 * wass_loss
            loss.backward()
            optimizer.step()
            # Append training statistics
            acc = (torch.argmax(logits, dim=1) ==
                   label).sum().item() / label.shape[0]
            train_acc.append(acc)
            train_loss.append(loss.detach().item())
            wass.append(wass_loss.detach().item())
            # print(f"Step {iter_count_train} | training acc: {acc} | wass: {-torch.log(-wass_loss)} | loss: {loss}")
            # pretrain_writer.add_scalar("Training/Accuracy", acc, iter_count_train)
            # pretrain_writer.add_scalar("Training/Loss", loss.detach().item(), iter_count_train)
            iter_count_train += 1

        # pretrain_writer.add_scalar("Training/Accuracy (Epoch)", np.mean(train_acc), epoch)
        # pretrain_writer.add_scalar("Training/Loss (Epoch)", np.mean(train_loss), epoch)
        print(f"Epoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
                | training acc: {np.mean(train_acc)} | Wass Loss {np.mean(wass)}")
        # Evaluation
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for idx, (img, label) in enumerate(ind_val_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                logits = model(img)
                loss = criterion(logits, label)
                acc = (torch.argmax(logits, dim=1) ==
                       label).sum().item() / label.shape[0]
                val_acc.append(acc)
                val_loss.append(loss.detach().item())
                # pretrain_writer.add_scalar("Training/Accuracy", acc, iter_count_val)
                # pretrain_writer.add_scalar("Training/Loss", loss.detach().item(), iter_count_val)
                iter_count_val += 1

            # pretrain_writer.add_scalar("Training/Accuracy (Epoch)", np.mean(val_acc), epoch)
            # pretrain_writer.add_scalar("Training/Loss (Epoch)", np.mean(val_loss), epoch)
            print(f"Epoch  # {epoch + 1} | validation loss: {np.mean(val_loss)} \
                | validation acc: {np.mean(val_acc)}")
    with torch.no_grad():
        # Evaluation
        torch.save(model.state_dict(),
                   log_dir + f"model-[{ood_bsz}]-[{max_epoch}]-[{mc}].pt")
        ic("Model Checkpoint Saved!")
        evaler.evaluate(model, f'mc={mc}', None,  False, ood)
        test_backbone_D(model, dset.ind_val_loader)
        mc_stop = time.time()
        ic(f"MC #{mc} time spent: {np.round(mc_stop - mc_start, 2)}s | About {np.round((mc_stop-mc_start)/60, 1)} mins")

# Display & save statistics
evaler.display_stats()
torch.save(evaler, log_dir + "eval.pt")
ic("EVALER & Stats saved successfully!")

stop = time.time()
ic(f"Training time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")
