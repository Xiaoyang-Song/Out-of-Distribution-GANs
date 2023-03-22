from traceback import StackSummary
from config import *
from dataset import MNIST, CIFAR10
from models.hparam import *
from utils import show_images, DIST_TYPE, get_dist_metric, Logger
from wass_loss import ood_wass_loss, ind_wass_loss
from models.gans import *
from metrics import *
from wasserstein import *
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


def ood_gan_g_loss(logits_fake, img_fake=None, img_ind=None,
                   img_ood=None, model=None):
    # 1. Wasserstein distance of G(z)
    w_fake = batch_wasserstein(logits_fake)
    # 2. Distance between X_in and G(z)
    # dist_fake_ind = get_dist_metric(
    #     img_fake, img_ind, ind_fake_sample_size, DIST_TYPE.EUC)

    ######################
    # Euclidean Distance #
    ######################
    # dist_fake_ind = torch.sqrt(torch.sum((xin - gz)**2))

    ###############
    # Correlation #
    ###############
    # xin = torch.mean(img_ind, dim=0).reshape(1, -1)  # 1 x HW
    # gz = torch.mean(img_fake, dim=0).reshape(1, -1)  # 1 x HW
    # target_mat = torch.cat([xin, gz])
    # dist_fake_ind = -torch.corrcoef(target_mat)[0][1]

    ########################
    # Feature Vector Angle #
    ########################
    target_feat = model(img_ood).mean(dim=0).squeeze()
    gz_feat = model(img_fake).mean(dim=0).squeeze()
    norm1, norm2 = LA.norm(target_feat), LA.norm(gz_feat)
    dist_fake_ood = torch.dot(target_feat, gz_feat) / (norm1 * norm2)
    dist_fake_ind = None
    return w_fake, dist_fake_ind, -dist_fake_ood


class OOD_GAN_TRAINER():
    def __init__(self, D, G, noise_dim,
                 bsz_tri, gd_steps_ratio, hp,
                 max_epochs,
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
        self.noise_dim = noise_dim
        self.dloss = ood_gan_d_loss
        self.gloss = ood_gan_g_loss
        # Training config
        self.bsz_tri = bsz_tri
        self.gd_steps_ratio = gd_steps_ratio
        self.max_epochs = max_epochs
        self.hp = hp

    def train(self, ind_loader, ood_img_batch, D_solver, G_solver, metric=None, pretrainedD=None, checkpoint=None):
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
            for steps, (x, y) in enumerate(ind_loader):
                cls = int(iter_count % 5)
                # cls=0
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                # Manually discard last batch
                if len(x) != self.bsz_tri:
                    continue
                # ---------------------- #
                # DISCRIMINATOR TRAINING #
                # ---------------------- #
                D_solver.zero_grad()
                # Logits for X_in
                logits_real = self.D(x)
                # Logits for G(z)
                seed = torch.rand(
                    (self.bsz_tri, self.noise_dim), device=DEVICE) * 2 - 1
                Gz = self.G(seed, [cls]*self.bsz_tri).detach()
                logits_fake = self.D(Gz)
                # Logits for X_ood
                ood_idx = np.random.choice(len(ood_img_batch), min(
                    len(ood_img_batch), 10), replace=False)
                ood_img = ood_img_batch[ood_idx, :, :, :]
                logits_ood = self.D(ood_img)
                # Compute loss
                ind_ce_loss, w_ood, w_fake = self.dloss(
                    logits_real, logits_fake, logits_ood, y)
                d_total = self.hp.ce * ind_ce_loss + \
                    self.hp.wass * (w_ood - (w_fake))

                # Write statistics
                self.writer.add_scalars("Discriminator Loss/each", {
                    'CE': ind_ce_loss.detach(),
                    'W_ood': -torch.log(-w_ood.detach()),
                    'W_z': -torch.log(-w_fake.detach())
                }, steps)
                self.writer.add_scalar(
                    "Discriminator Loss/total", d_total.detach(), steps)

                # Update
                d_total.backward()
                D_solver.step()

                # ------------------ #
                # GENERATOR TRAINING #
                # ------------------ #
                for g_step in range(self.gd_steps_ratio):
                    G_solver.zero_grad()
                    # Logits for G(z)
                    seed = torch.rand(
                        (self.bsz_tri, self.noise_dim), device=DEVICE) * 2 - 1
                    Gz = self.G(seed,  [cls]*self.bsz_tri).to(DEVICE).detach()
                    logits_fake = self.D(Gz)
                    # Compute loss
                    # Minimize ood and gz
                    w_z, _, dist_fake_ood = self.gloss(
                        logits_fake, Gz, None, ood_img_batch, model=metric)
                    g_total = -(self.hp.wass * (w_z) -
                                self.hp.dist * dist_fake_ood)

                    # Write statistics
                    global_step = steps*self.gd_steps_ratio+g_step
                    self.writer.add_scalars("Generator Loss/each", {
                        'W_z': w_z.detach(),
                        'd_ood': -dist_fake_ood.detach()
                    }, global_step)
                    self.writer.add_scalar(
                        f"Generator Loss/distance/{cls}", -dist_fake_ood.detach(), global_step // 5)
                    self.writer.add_scalar(
                        "Generator Loss/total", g_total.detach(), global_step)

                    # Update
                    g_total.backward()
                    G_solver.step()

                # Print out statistics
                if (iter_count % self.n_steps_log == 0):
                    # print(
                    #     f"Step: {steps:<4} | {cls} | D: {d_total.item(): .4f} | CE: {ind_ce_loss.item(): .4f} | W_OoD: {-torch.log(-w_ood).item(): .4f} | W_z: {-torch.log(-w_fake).item(): .4f} | G: {g_total.item(): .4f} | d_ind: {dist_fake_ind.item(): .4f} | d_pixel: {((dist_fake_ind.item()**2)/784)**0.5: .4f} | W_z: {-torch.log(-w_z).item(): .4f}")
                    print(
                        f"Step: {steps:<4} | {cls} | D: {d_total.item(): .4f} | CE: {ind_ce_loss.item(): .4f} | W_OoD: {-torch.log(-w_ood).item(): .4f} | W_z: {-torch.log(-w_fake).item(): .4f} | G: {g_total.item(): .4f} | CosSim: {-dist_fake_ood.item(): .4f} | W_z: {-torch.log(-w_z).item(): .4f}")
                iter_count += 1

            # Save checkpoint
            ckpt_name = f"{self.ckpt_dir}{self.ckpt_name}_[{epoch}].pt"
            torch.save({
                'D-state': self.D.state_dict(),
                'G-state': self.G.state_dict()
                # 'writer': self.writer
            }, ckpt_name)
            print(f'New checkpoint created at the end of epoch {epoch}.')
