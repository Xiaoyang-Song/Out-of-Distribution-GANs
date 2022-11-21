from config import *
from dataset import MNIST, CIFAR10
from models.hparam import *
from utils import show_images, DIST_TYPE, get_dist_metric, Logger
from wass_loss import ood_wass_loss, ind_wass_loss
from models.gans import *
from metrics import *
from wasserstein import *
from time import gmtime, strftime


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
                   img_ood=None, ind_fake_sample_size=64):
    # 1. Wasserstein distance of G(z)
    w_fake = batch_wasserstein(logits_fake)
    # 2. Distance between X_in and G(z)
    dist_fake_ind = get_dist_metric(
        img_fake, img_ind, ind_fake_sample_size, DIST_TYPE.EUC)
    # 3. Distance between X_ood and G(z)
    dist_fake_ood = None
    return w_fake, dist_fake_ind, dist_fake_ood


class OOD_GAN_TRAINER():
    def __init__(self):
        super().__init__()


def ood_gan_trainer(ind_loader, ood_loader, D, G, D_solver, G_solver, discriminator_loss,
                    generator_loss, metric, img_info, backbone=GAN_BACKBONE.FC, checkpoint=None, checkpoint_save_addr=None, hp=HParam(),
                    g_d_ratio=1, save_filename=None, show_every=250, pretrained_D=None,
                    batch_size=128, noise_size=96, num_epochs=10, logger=None, logger_max_iter=None):
    # Assertion Check of img_info
    assert img_info is not None, 'Expect img_info to be a dictionary containing H, W, and C.'
    H, W, C = img_info['H'], img_info['W'], img_info['C']
    if pretrained_D is not None:
        pretrain = torch.load(pretrained_D)
        D.load_state_dict(pretrain['model_state_dict'])
        print("Pretrained D state is loaded.")
    if checkpoint is not None:
        chpt = torch.load(checkpoint['addr'])
        D.load_state_dict(chpt['D-state'])
        G.load_state_dict(chpt['G-state'])
        print(f"Checkpoint [{checkpoint['id']} loaded.")

    # Assertion Check of Logger arguments
    assert (logger_max_iter is None and logger is None) or (
        logger_max_iter is not None and logger is not None), \
        'Expect logger and logger_iter to be not None or None simultaneously.'
    # OBTAIN OOD SAMPLES
    # assert ood_loader is not None, 'Expect ood_loader to be not None.'
    ood_img_batch, ood_img_batch_label = next(iter(ood_loader))

    iter_count = 0
    for epoch in range(num_epochs):
        for x, y in ind_loader:
            if len(x) != batch_size:
                continue
            # EARLY STOP FOR SAMPLE TRAINING WITH Logger
            if iter_count >= logger_max_iter:
                print(
                    f'Sample Training ({iter_count} iterations) with Logger Finished.')
                return

            # Discriminator Training
            D_solver.zero_grad()
            # TODO: Revise backbone architecture to make sure it works for unflattened images.
            real_data = x.view(-1, C*H*W).to(DEVICE)  # B x HWC
            # logits_real = D(2 * (real_data - 0.5))
            logits_real = D(2 * (x - 0.5))

            num_trial = 0
            while True:
                g_fake_seed = sample_noise(
                    batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
                fake_images = G(g_fake_seed).detach()
                if satisfied():
                    print(
                        f'[{iter_count}] Trial {num_trial} succeeds. Training resumes.')
                    break
                num_trial += 1
                print(f'Trial {num_trial} fails. Resampling...')
            logits_fake = D(fake_images)
            # TODO: decouple OOD GANs and the Original GANs in this script
            # ood_imgs = ood_img_batch.view(-1, H*W*C).to(DEVICE)
            logits_ood = D(ood_img_batch)
            ind_ce_loss, w_ood, w_fake = discriminator_loss(logits_real, logits_fake, logits_ood=logits_ood,
                                                            labels_real=y, gan_type=GAN_TYPE.OOD)

            # print("Discriminator Loss Terms:")
            # ic(ind_ce_loss)
            # ic(zsl_ood)
            # ic(zsl_fake)
            d_total_error = hp.ce * ind_ce_loss + \
                hp.wass * (w_ood - (w_fake))
            if logger is not None:
                logger.ap_d_ls(ind_ce_loss, -w_ood, -w_fake)

            d_total_error.backward()
            D_solver.step()

            # Generator Training
            for num_g_steps in range(g_d_ratio):
                G_solver.zero_grad()
                g_fake_seed = sample_noise(
                    batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
                # fake_images = G(g_fake_seed).view(
                #     (-1, C, H, W)).to(DEVICE)
                fake_images = G(g_fake_seed).to(DEVICE)

                gen_logits_fake = D(fake_images)
                # zsl_fake, dist_fake_ind, dist_fake_ood = generator_loss(
                #     gen_logits_fake, fake_images, ood_imgs, real_data, gan_type=GAN_TYPE.OOD)
                w_fake_g, dist_fake_ind, dist_fake_ood = generator_loss(
                    gen_logits_fake, fake_images, x, ood_img_batch, dist=metric, gan_type=GAN_TYPE.OOD)
                # print("Generator Loss Terms:")
                # ic(zsl_fake)
                # ic(dist_fake_ind)
                # ic(dist_fake_ood)

                # g_total_error = -(hp.wass * (-zsl_fake) +
                #                   hp.dist * (-dist_fake_ind + dist_fake_ood))
                # Only Ind distance
                g_total_error = -(hp.wass * (w_fake_g) +
                                  hp.dist * (dist_fake_ind))
                # No distance
                # g_total_error = -(hp.wass * (-zsl_fake))
                if logger is not None:
                    # logger.ap_g_ls(
                    #     -zsl_fake, dist_fake_ind, -dist_fake_ood)
                    logger.ap_g_ls(
                        -w_fake_g, dist_fake_ind, torch.tensor(0))
                g_total_error.backward()
                G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(
                    iter_count, d_total_error.item(), g_total_error.item()))
                imgs_numpy = fake_images.data.cpu()  # .numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1

        if checkpoint_save_addr is not None:
            print(f'New checkpoint created at the end of epoch {epoch}.')
            chpt_name = checkpoint_save_addr + get_time_signature() + '.pt'
            torch.save({
                'D-state': D.state_dict(),
                'G-state': G.state_dict(),
                'logger': logger
            }, chpt_name)
