from config import *
from dataset import MNIST, CIFAR10
from models.hparam import *
from wass_loss import ood_wass_loss, ind_wass_loss
from wasserstein import *

BATCH_SIZE = 128  # for training
TEST_BATCH_SIZE = 64  # for testing (not used for now)
NUM_CLASSES = 10  # TODO: This should be done automatically in the future


class GAN_TYPE(Enum):
    NAIVE, OOD = list(range(2))


class GAN_BACKBONE(Enum):
    FC, CONV = list(range(2))


def zero_softmax_loss(x): return torch.log(ood_wass_loss(
    torch.softmax(x, dim=-1), NUM_CLASSES)).mean()


def sample_noise(batch_size, noise_dim, extra_dim=False, dtype=torch.float, device=DEVICE):
    if extra_dim:
        return torch.rand((batch_size, noise_dim, 1, 1), device=device) * 2 - 1
    else:
        noise = torch.rand((batch_size, noise_dim), device=device) * 2 - 1
        return noise


def discriminator(gan_type=GAN_TYPE.NAIVE):
    assert gan_type is GAN_TYPE.NAIVE or GAN_TYPE.OOD, 'Expect gan_type to be one of GAN_TYPE.'
    model = [
        nn.Linear(784, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01)
    ]
    out_dim = 10 if gan_type == GAN_TYPE.OOD else 1
    model.append(nn.Linear(256, out_dim))
    return nn.Sequential(*model)


def generator(noise_dim=NOISE_DIM):
    # TODO: Implement Deep-Convolutional GANs
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return model


def discriminator_loss(logits_real, logits_fake, logits_ood=None,
                       labels_real=None, gan_type=GAN_TYPE.NAIVE):
    # TODO: decouple two different GANs
    if gan_type == GAN_TYPE.NAIVE:
        label = torch.ones_like(logits_fake, dtype=logits_real.dtype)
        rl_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_real, label, reduction='none').mean()
        fk_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_fake, label - 1, reduction='none').mean()
        loss = rl_loss + fk_loss
        return loss
    elif gan_type == GAN_TYPE.OOD:
        assert logits_ood is not None, 'Expect logits_ood to be not None.'
        assert labels_real is not None, 'Expect labels_real to be not None.'
        # Compute discriminator loss
        criterion = nn.CrossEntropyLoss()
        ind_ce_loss = criterion(logits_real, labels_real)
        # Compute wass_loss term
        assert logits_ood.requires_grad
        assert logits_fake.requires_grad
        w_ood = batch_wasserstein(logits_ood)
        w_fake = batch_wasserstein(logits_fake)
        return ind_ce_loss, w_ood, w_fake
    else:
        assert False, 'Unrecognized GAN_TYPE.'


def generator_loss(logits_fake, img_fake=None, img_ind=None,
                   img_ood=None, dist_sample_size=64, dist=None, gan_type=GAN_TYPE.NAIVE):
    if gan_type == GAN_TYPE.NAIVE:
        label = torch.ones_like(logits_fake, dtype=logits_fake.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_fake, label, reduction='none').mean()
        return loss
    elif gan_type == GAN_TYPE.OOD:
        assert img_fake is not None, 'Expect img_fake to be not None.'
        assert img_ood is not None, 'Expect img_ood to be not None.'
        assert img_ind is not None, 'Expect img_ind to be not None.'
        # Compute generator loss for OOD GANs
        w_fake_g = batch_wasserstein(logits_fake)
        # x = torch.repeat_interleave(x, 3, 1)
        dist_fake_ind = dist(torch.repeat_interleave(
            img_ind, 3, 1), torch.repeat_interleave(img_fake, 3, 1))
        # dist_fake_ood = dist(img_ood, img_fake)
        dist_fake_ood = None
        # dist_fake_ind = get_dist_metric(
        #     img_fake, img_ind, dist_sample_size, DIST_TYPE.COR)
        # dist_fake_ood = get_dist_metric(
        #     img_fake, img_ood, dist_sample_size, DIST_TYPE.COR)
        return w_fake_g, dist_fake_ind, dist_fake_ood
    else:
        assert False, 'Unrecognized GAN_TYPE.'


def get_optimizer(model):
    # TODO: Make this more generic later.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer

# TODO: Add more vectorization and batch processing to make OOD GAN more efficient.


def gan_trainer(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, hp=HParam(), g_d_ratio=1, save_filename=None, gan_type=GAN_TYPE.NAIVE, show_every=250,
                batch_size=128, noise_size=96, num_epochs=10, ood_loader=None, ood_img_batch_size=BATCH_SIZE,
                ood_img_sample=None, logger=None, logger_max_iter=None):
    # Assertion Check of GD Loss Tracker arguments
    assert (logger_max_iter is None and logger is None) or (
        logger_max_iter is not None and logger is not None), \
        'Expect logger and logger_iter to be not None or None simultaneously.'
    # OBTAIN OOD SAMPLES
    if gan_type == GAN_TYPE.OOD:
        if ood_loader != None:
            ood_img_batch, ood_img_batch_label = next(iter(ood_loader))
        else:
            assert ood_img_sample != None, 'Please specify ood image sample when training OOD GANs.'
            _, _, ood_tri_loader, _ = ood_img_sample(
                ood_img_batch_size, TEST_BATCH_SIZE)
            ood_img_batch, ood_img_batch_label = next(iter(ood_tri_loader))
            ic(ood_img_batch.shape)
            assert type(
                ood_img_batch) == torch.Tensor, 'Expect the image batch to be a torch tensor.'
            # ic(ood_img_batch.shape)  # 128 x 3 x 28 x 28
            # TODO: This portion of code only works for CIFAR10 and MNIST transformation
            # TODO: This portion of code MUST be rewritten in the future.
            ood_img_batch = torch.mean(ood_img_batch, dim=1)
            # ic(ood_img_batch.shape)  # (128, 28, 28) OR (B, H, W)
            # ic(ood_img_batch_label.shape)  # (128,) OR (B,)
    iter_count = 0
    for epoch in range(num_epochs):
        for x, y in loader_train:
            # x: (B, 28, 28) for
            if len(x) != batch_size:
                continue
            # EARLY STOP FOR SAMPLE TRAINING WITH GD_LOSS_TRACKER
            if iter_count >= logger_max_iter:
                print('Sample Training with Logger Finished.')
                return
            # Discriminator Training
            D_solver.zero_grad()
            # TODO: Revise backbone architecture to make sure it works for unflattened images.
            real_data = x.view(-1, 784).to(DEVICE)  # B x 784
            logits_real = D(2 * (real_data - 0.5))

            g_fake_seed = sample_noise(
                batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)
            # TODO: decouple OOD GANs and the Original GANs in this script
            if gan_type == GAN_TYPE.OOD:
                ood_imgs = ood_img_batch.view(-1, 784).to(DEVICE)
                logits_ood = D(ood_imgs)
                ind_ce_loss, zsl_ood, zsl_fake = discriminator_loss(logits_real, logits_fake, logits_ood=logits_ood,
                                                                    labels_real=y, gan_type=GAN_TYPE.OOD)
                d_total_error = hp.ce * ind_ce_loss - \
                    hp.wass * (zsl_ood + zsl_fake)
                if logger is not None:
                    logger.ap_d_ls(ind_ce_loss, -zsl_ood, -zsl_fake)
            else:
                d_total_error = discriminator_loss(logits_real, logits_fake)

            d_total_error.backward()
            D_solver.step()

            # Generator Training
            for num_g_steps in range(g_d_ratio):
                G_solver.zero_grad()
                g_fake_seed = sample_noise(
                    batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(fake_images)

                if gan_type == GAN_TYPE.OOD:
                    zsl_fake, dist_fake_ind, dist_fake_ood = generator_loss(
                        gen_logits_fake, fake_images, ood_imgs, real_data, gan_type=GAN_TYPE.OOD)
                    g_total_error = - hp.wass * zsl_fake + \
                        hp.dist * (-dist_fake_ind + dist_fake_ood)
                    if logger is not None:
                        logger.ap_g_ls(
                            zsl_fake, dist_fake_ind, dist_fake_ood)
                else:
                    g_total_error = generator_loss(gen_logits_fake)
                g_total_error.backward()
                G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(
                    iter_count, d_total_error.item(), g_total_error.item()))
                imgs_numpy = fake_images.data.cpu()  # .numpy()
                # show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
        if epoch == num_epochs - 1:
            # show_images(imgs_numpy[0:16])
            if save_filename is not None:
                plt.savefig(os.path.join(GAN_SAVE_PATH, save_filename))


if __name__ == "__main__":
    ic("Hello gans.py")
    ic(f"Device: {DEVICE}")
    # TEST GAN TRAINER
    mnist_tri_set, mnist_val_set, mnist_tri_loader, mnist_val_loader = MNIST(
        128, 32, 2, True)
    D = discriminator(gan_type=GAN_TYPE.OOD).to(DEVICE)
    G = generator().to(DEVICE)
    D_solver = get_optimizer(D)
    G_solver = get_optimizer(G)
    # TODO: Maybe a problematic test code depends on machine, Fix this later
    gan_trainer(mnist_tri_loader, D, G, D_solver, G_solver,
                discriminator_loss, generator_loss, 'fc_gan_results.jpg',
                gan_type=GAN_TYPE.OOD, ood_img_sample=CIFAR10)
