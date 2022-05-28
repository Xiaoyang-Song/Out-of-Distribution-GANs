from config import *
from utils import show_images


NOISE_DIM = 96


def sample_noise(batch_size, noise_dim, dtype=torch.float, device=DEVICE):
    noise = torch.rand((batch_size, noise_dim), device=device) * 2 - 1
    return noise


def discriminator():
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.01),
        nn.Linear(256, 1)
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


def discriminator_loss(logits_real, logits_fake, type='ind'):
    if type == 'ind':
        label = torch.ones_like(logits_fake, dtype=logits_real.dtype)
        rl_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_real, label, reduction='none').mean()
        fk_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_fake, label - 1, reduction='none').mean()
        loss = rl_loss + fk_loss
        return loss
    else:
        return None


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
                generator_loss, save_filename, type='ind', show_every=250,
                batch_size=128, noise_size=96, num_epochs=10, loader_train_ood=None):
    # assert type == 'ind' and loader_train_ood != None
    iter_count = 0
    for epoch in range(num_epochs):
        for x, y in loader_train:
            # for ind_loader, ood_loader in zip(loader_train, loader_train_ood):
            #     x, y = ind_loader
            #     ood_x, ood_y = ood_loader
            if len(x) != batch_size:
                continue

            # Discriminator Training
            D_solver.zero_grad()
            real_data = x.view(-1, 784).to(DEVICE)
            logits_real = D(2 * (real_data - 0.5))

            g_fake_seed = sample_noise(
                batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)

            # ood_data = ood_x.view(-1, 784).to(DEVICE)
            # logits_ood = D(ood_data)

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
            plt.savefig(os.path.join(GAN_SAVE_PATH, save_filename))


if __name__ == "__main__":
    ic("Hello gans.py")
    ic(f"Device: {DEVICE}")
