from config import *
from dataset import MNIST, CIFAR10
from utils import show_images, DIST_TYPE, get_dist_metric
from wass_loss import ood_wass_loss, ind_wass_loss


NOISE_DIM = 96
BATCH_SIZE = 128  # for training
TEST_BATCH_SIZE = 64  # for testing (not used for now)
NUM_CLASSES = 10  # TODO: This should be done automatically in the future


class GAN_TYPE(Enum):
    NAIVE, OOD = list(range(2))


def sample_noise(batch_size, noise_dim, dtype=torch.float, device=DEVICE):
    noise = torch.rand((batch_size, noise_dim), device=device) * 2 - 1
    return noise


def discriminator(gan_type=GAN_TYPE.NAIVE):
    if gan_type == GAN_TYPE.NAIVE:
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)
        )
    elif gan_type == GAN_TYPE.OOD:
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 10)
        )
    return model


def generator(noise_dim=NOISE_DIM):
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
        # TODO: current implementation is not numerically stable; change this later.
        def zero_softmax_loss(x): return torch.log(ood_wass_loss(
            torch.softmax(x), NUM_CLASSES)).mean()
        zsl_ood, zsl_fake = [zero_softmax_loss(
            logit) for logit in (logits_ood, logits_fake)]
        return ind_ce_loss + zsl_ood + zsl_fake
    else:
        assert False, 'Unrecognized GAN_TYPE.'


def generator_loss(logits_fake, type='ind'):
    if type == 'ind':
        label = torch.ones_like(logits_fake, dtype=logits_fake.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_fake, label, reduction='none').mean()
        return loss
    else:
        return None


def get_optimizer(model):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer


def gan_trainer(loader_train, D, G, D_solver, G_solver, discriminator_loss,
                generator_loss, save_filename=None, gan_type=GAN_TYPE.NAIVE, show_every=250,
                batch_size=128, noise_size=96, num_epochs=10, ood_img_batch_size=BATCH_SIZE,
                ood_img_sample=None):
    # OBTAIN OOD SAMPLES
    if gan_type == GAN_TYPE.OOD:
        assert ood_img_sample != None, 'Please specify ood image sample when training OOD GANs.'
        _, _, ood_tri_loader, _ = ood_img_sample(
            ood_img_batch_size, TEST_BATCH_SIZE)
        # sample_idx = torch.randint(0, len(ood_tri_loader), (ood_img_batch_size,))
        # ic(len(sample_idx))
        ood_img_batch, ood_img_batch_label = next(iter(ood_tri_loader))
        assert type(
            ood_img_batch) == torch.Tensor, 'Expect the image batch to be a torch tensor.'
        # ic(ood_img_batch.shape)  # 128 x 3 x 28 x 28
        # TODO: This portion of code only works for CIFAR10 and MNIST transformation
        ood_img_batch = torch.mean(ood_img_batch, dim=1)
        # ic(ood_img_batch.shape)  # (128, 28, 28) OR (B, H, W)
        # ic(ood_img_batch_label.shape)  # (128,) OR (B,)
    iter_count = 0
    for epoch in range(num_epochs):
        for x, y in loader_train:
            # x: (B, 28, 28)
            # for ind_loader, ood_loader in zip(loader_train, loader_train_ood):
            #     x, y = ind_loader
            #     ood_x, ood_y = ood_loader
            if len(x) != batch_size:
                continue

            # Discriminator Training
            D_solver.zero_grad()
            real_data = x.view(-1, 784).to(DEVICE)  # B x 784
            logits_real = D(2 * (real_data - 0.5))

            g_fake_seed = sample_noise(
                batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)

            ood_imgs = ood_img_batch.view(-1, 784).to(DEVICE)
            logits_ood = D(ood_imgs)

            test_d_total_error = discriminator_loss(logits_real, logits_fake, logits_ood=logits_ood,
                                                    labels_real=y, gan_type=GAN_TYPE.OOD)
            ic(test_d_total_error)
            return

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            # Generator Training
            G_solver.zero_grad()
            g_fake_seed = sample_noise(
                batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images)

            # OOD Generator
            # ood_dist = torch.mean(ood_data, dim=0).squeeze()  # (784,)

            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(
                    iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu()  # .numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
        if epoch == num_epochs - 1:
            show_images(imgs_numpy[0:16])
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
    gan_trainer(mnist_tri_loader, D, G, D_solver, G_solver,
                discriminator_loss, generator_loss, 'fc_gan_results.jpg',
                gan_type=GAN_TYPE.OOD, ood_img_sample=CIFAR10)