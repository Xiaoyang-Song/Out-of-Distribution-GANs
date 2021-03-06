from numpy import append
from config import *
from models.gans import GAN_TYPE


def dc_discriminator(img_info=None, gan_type=GAN_TYPE.NAIVE):
    H, W, C = img_info['H'], img_info['W'], img_info['C']
    assert gan_type is GAN_TYPE.NAIVE or GAN_TYPE.OOD, 'Expect gan_type to be one of GAN_TYPE.'
    model = [
        nn.Unflatten(1, (C, H, W)),
        nn.Conv2d(C, 32, 5),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2),
        nn.Flatten(1, -1),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(0.01)
    ]
    out_dim = 10 if gan_type == GAN_TYPE.OOD else 1
    model.append(nn.Linear(4 * 4 * 64, out_dim))
    return nn.Sequential(*model)


def dc_generator(noise_dim=NOISE_DIM):
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(7 * 7 * 128),
        nn.Unflatten(-1, (128, 7, 7)),
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        nn.Tanh(),
        nn.Flatten(1, -1)
    )
    return model


if __name__ == '__main__':
    # SANITY CHECK
    dc_disc = dc_discriminator(gan_type=GAN_TYPE.OOD)
    ic(dc_disc)
