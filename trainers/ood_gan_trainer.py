from config import *
from dataset import MNIST, CIFAR10
from models.hparam import *
from utils import show_images, DIST_TYPE, get_dist_metric, Logger
from wass_loss import ood_wass_loss, ind_wass_loss
from models.gans import *

BATCH_SIZE = 128  # for training
TEST_BATCH_SIZE = 64  # for testing (not used for now)
NUM_CLASSES = 10  # TODO: This should be done automatically in the future


def load_checkpoint():
    return None, None


def satisfied():
    return True


def ood_gan_trainer(ind_loader, ood_loader, D, G, D_solver, G_solver, discriminator_loss,
                    generator_loss, img_info, backbone=GAN_BACKBONE.FC, checkpoint=None, checkpoint_save_addr=None, hp=HParam(),
                    g_d_ratio=1, save_filename=None, show_every=250,
                    batch_size=128, noise_size=96, num_epochs=10, logger=None, logger_max_iter=None):
    # Assertion Check of img_info
    assert img_info is not None, 'Expect img_info to be a dictionary containing H, W, and C.'
    H, W, C = img_info['H'], img_info['W'], img_info['C']

    if checkpoint is not None:
        # TODO: Implement this function later
        D, G = load_checkpoint()

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
            logits_real = D(2 * (real_data - 0.5))

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
            ood_imgs = ood_img_batch.view(-1, H*W*C).to(DEVICE)
            logits_ood = D(ood_imgs)
            ind_ce_loss, zsl_ood, zsl_fake = discriminator_loss(logits_real, logits_fake, logits_ood=logits_ood,
                                                                labels_real=y, gan_type=GAN_TYPE.OOD)

            # print("Discriminator Loss Terms:")
            # ic(ind_ce_loss)
            # ic(zsl_ood)
            # ic(zsl_fake)
            d_total_error = hp.ce * ind_ce_loss + \
                hp.wass * (-zsl_ood - zsl_fake)
            if logger is not None:
                logger.ap_d_ls(ind_ce_loss, -zsl_ood, -zsl_fake)

            d_total_error.backward()
            D_solver.step()

            # Generator Training
            for num_g_steps in range(g_d_ratio):
                G_solver.zero_grad()
                g_fake_seed = sample_noise(
                    batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(fake_images)
                zsl_fake, dist_fake_ind, dist_fake_ood = generator_loss(
                    gen_logits_fake, fake_images, ood_imgs, real_data, gan_type=GAN_TYPE.OOD)
                # print("Generator Loss Terms:")
                # ic(zsl_fake)
                # ic(dist_fake_ind)
                # ic(dist_fake_ood)

                g_total_error = -(hp.wass * (-zsl_fake) +
                                  hp.dist * (dist_fake_ind - dist_fake_ood))
                if logger is not None:
                    logger.ap_g_ls(
                        -zsl_fake, dist_fake_ind, -dist_fake_ood)
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

        # TODO: Checkpointing
        print(f'Checkpoint created at the end of epoch {epoch}.')

        # TODO: Change GAN_SAVE_PATH
        if epoch == num_epochs - 1:
            show_images(imgs_numpy[0:16])
            if save_filename is not None:
                plt.savefig(os.path.join(GAN_SAVE_PATH, save_filename))


if __name__ == '__main__':
    ic("Hello ood_gan_trainer.py")
