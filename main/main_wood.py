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
ic(ood_bsz)
log_dir = f"../checkpoint/MNIST/WOOD/Test/{ood_bsz}/"
ckpt_dir = f"../checkpoint/MNIST/WOOD/Test/{ood_bsz}/"
pretrained_dir = f"../checkpoint/pretrained/mnist/"
# pretrained_dir = f"../checkpoint/pretrained/mnist/"
##### Hyperparameters #####
img_info = {'H': 28, 'W': 28, 'C': 1}
max_epoch = args.num_epochs
##### Dataset #####
dset = DSET('mnist', 50, 256, [2, 3, 6, 8, 9], [1, 7])
evaler = EVALER(dset.ind_train, dset.ind_val, dset.ood_val, ood_bsz, log_dir)

##### Monte Carlo config #####
MC_NUM = args.mc

for mc in range(MC_NUM):
    mc_start = time.time()
    ic(f"Monte Carlo Iteration {mc}")
    ##### logging information #####
    writer_name = log_dir + f"MNIST-WOOD-[{ood_bsz}]-[{mc}]"
    ckpt_name = f'MNIST-WOOD-[{ood_bsz}]-balanced-[{mc}]'

    model = DC_D(5, img_info).to(DEVICE)
    ckpt = torch.load(pretrained_dir + "mnist-[23689]-D.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # Training dataset
    ind_tri_loader = dset.ind_train_loader
    ind_val_loader = dset.ind_val_loader
    if args.balanced == 'balance':
        ic('Balanced Experiment')
        ood_img_batch, ood_img_label = dset.ood_sample(ood_bsz, 'balanced')
    else:
        ic("Imbalanced Experiment")
        ood_img_batch, ood_img_label = dset.ood_sample(
            ood_bsz, 'imbalanced', [0])
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
        # break
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
        # print(f"Epoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
        #         | training acc: {np.mean(train_acc)} | Wass Loss {np.mean(wass)}")
        # Evaluation
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
    # Evaluation
    evaler.compute_stats(model, f'mc={mc}', None,  True, [1, 7])
    torch.save(model.state_dict(),
               log_dir + f"model-[{ood_bsz}]-[{mc}].pt")
    mc_stop = time.time()
    ic(f"MC #{mc} time spent: {np.round(mc_stop - mc_start, 2)}s | About {np.round((mc_stop-mc_start)/60, 1)} mins")

# Display & save statistics
evaler.display_stats()
torch.save(evaler, log_dir + "eval.pt")
ic("EVALER & Stats saved successfully!")

stop = time.time()
ic(f"Training time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")
