from traceback import StackSummary
from config import *
from dataset import MNIST, CIFAR10
from models.hparam import *
from utils import show_images, DIST_TYPE, get_dist_metric, Logger
from wass_loss import ood_wass_loss, ind_wass_loss
from models.gans import *
from metrics import *
from wasserstein import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from sklearn.decomposition import PCA


def ood_gan_d_loss(logits_real, logits_fake, logits_ood, labels_real):
    # 1: CrossEntropy of X_in
    criterion = nn.CrossEntropyLoss()
    ind_ce_loss = criterion(logits_real, labels_real)
    # 2. W_ood
    assert logits_ood.requires_grad
    w_ood = batch_wasserstein(logits_ood)
    # 3. W_z
    assert logits_fake.requires_grad
    w_fake = batch_wasserstein(logits_fake)
    return ind_ce_loss, w_ood, w_fake


def ood_gan_g_loss(logits_fake, gz, xood):
    # 1. Wasserstein distance of G(z)
    assert logits_fake.requires_grad
    w_fake = batch_wasserstein(logits_fake)
    # distance term
    # print(torch.mean(gz, dim=0) - torch.mean(xood, dim=0))
    dist = torch.sqrt(torch.sum((torch.mean(gz) - torch.mean(xood))**2))
    assert dist.requires_grad
    return w_fake, dist

class OOD_GAN_TRAINER():
    def __init__(self, D, G, noise_dim, num_classes,
                 bsz_tri, g_steps_ratio, d_steps_ratio, 
                 scaling, hp, max_epochs, ood_bsz,
                 writer_name, ckpt_name, ckpt_dir,
                 n_steps_show=100, n_steps_log=1):
        super().__init__()
        # Logger information
        self.writer_name = writer_name
        self.writer = SummaryWriter(writer_name)
        self.ckpt_name = ckpt_name
        self.ckpt_dir = ckpt_dir
        # Print statement config
        self.n_steps_show = n_steps_show
        self.n_steps_log = n_steps_log
        # Backbone models & info
        self.D = D
        self.G = G
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.dloss = ood_gan_d_loss
        self.gloss = ood_gan_g_loss
        # Training config
        self.bsz_tri = bsz_tri
        self.d_steps_ratio = d_steps_ratio
        self.g_steps_ratio = g_steps_ratio
        self.scaling = scaling
        self.max_epochs = max_epochs
        self.hp = hp
        self.ood_bsz = ood_bsz

    def train(self, ind_loader, ood_img_batch, D_solver, G_solver, pretrainedD=None, checkpoint=None):
        # Load pretrained Discriminator
        if pretrainedD is not None:
            pretrain = torch.load(pretrainedD)
            self.D.load_state_dict(pretrain['model_state_dict'])
            print("Pretrained D model state is loaded.")
        # Load checkpoint information
        if checkpoint is not None:
            ckpt = torch.load(checkpoint['addr'])
            self.D.load_state_dict(ckpt['D-state'])
            self.G.load_state_dict(ckpt['G-state'])
            self.writer = ckpt['writer']
            print(f"Checkpoint [{checkpoint['id']} loaded.")
        # Print out OoD sample statistics
        print(f"OoD sample shape: {ood_img_batch.shape}")
        ood_img_batch = ood_img_batch.to(DEVICE)
        iter_count = 0
        for epoch in range(self.max_epochs):
            for steps, (x, y) in enumerate(tqdm(ind_loader)):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                # Manually discard last batch
                if len(x) != self.bsz_tri:
                    continue
                # ---------------------- #
                # DISCRIMINATOR TRAINING #
                # ---------------------- #
                for dstep in range(self.d_steps_ratio):
                    D_solver.zero_grad()
                    # Logits for X_in
                    logits_real = self.D(x)

                    seed = torch.rand((self.bsz_tri, self.noise_dim, 1, 1), device=DEVICE)

                    Gz = self.G(seed)
                    logits_fake = self.D(Gz)
                    # Logits for X_ood
                    ood_idx = np.random.choice(len(ood_img_batch), min(
                        len(ood_img_batch), self.ood_bsz), replace=False)
                    ood_img = ood_img_batch[ood_idx, :, :, :].to(DEVICE)
                    logits_ood = self.D(ood_img)

                    # Compute loss
                    ind_ce_loss, w_ood, w_fake = self.dloss(
                        logits_real, logits_fake, logits_ood, y)
                    d_total = self.hp.ce * ind_ce_loss - self.hp.wass * (w_ood - w_fake * self.scaling)
                    
                    # Write statistics
                    global_step_d = steps * self.d_steps_ratio + dstep
                    self.writer.add_scalars("Discriminator Loss/each", {
                        'CE': ind_ce_loss.detach(),
                        'W_ood': w_ood.detach(),
                        'W_z': w_fake.detach()
                    }, global_step_d)
                    self.writer.add_scalar(
                        "Discriminator Loss/total", d_total.detach(), global_step_d)
                    
                    # Update
                    d_total.backward()
                    D_solver.step()


                # ------------------ #
                # GENERATOR TRAINING #
                # ------------------ #
                for gstep in range(self.g_steps_ratio):
                    G_solver.zero_grad()
                    # Logits for G(z)
                    seed = torch.rand( (self.bsz_tri, self.noise_dim, 1, 1), device=DEVICE)

                    Gz = self.G(seed)
                    logits_fake = self.D(Gz)

                    w_z, dist = ood_gan_g_loss(logits_fake, Gz, ood_img)
                    # g_total = -w_wass * (w_z) + dist * w_dist
                    g_total = -self.hp.wass * w_z * self.scaling

                    # Write statistics
                    global_step_g = steps * self.g_steps_ratio + gstep

                    self.writer.add_scalars("Generator Loss/each", {
                        'W_z': w_z.detach()
                    }, global_step_g)
                    self.writer.add_scalar(
                        "Generator Loss/total", g_total.detach(), global_step_g)

                    # Update
                    g_total.backward()
                    G_solver.step()

                # Print out statistics
                if (iter_count % self.n_steps_log == 0):
                    print(
                        f"Step: {steps:<4} | D: {d_total.item(): .4f} | CE: {ind_ce_loss.item(): .4f} | W_OoD: {w_ood.item(): .4f} | W_z: {w_fake.item(): .4f} | G: {g_total.item(): .4f} | W_z: {w_z.item(): .4f} | dist: {dist:.4f}")
                iter_count += 1

            # Save checkpoint
            ckpt_name = f"{self.ckpt_dir}{self.ckpt_name}_[{epoch}].pt"
            torch.save({
                'D-state': self.D.state_dict(),
                'G-state': self.G.state_dict()
                # 'writer': self.writer
            }, ckpt_name)
            print(f'New checkpoint created at the end of epoch {epoch}.')
