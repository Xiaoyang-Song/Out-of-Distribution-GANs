from config import *


def dc_discriminator():
    model = nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        nn.Conv2d(1, 32, 5),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 5),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2),
        nn.Flatten(1, -1),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.LeakyReLU(0.01),
        nn.Linear(4 * 4 * 64, 1)
    )
    return model


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
