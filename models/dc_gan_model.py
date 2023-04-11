from config import *


class D(nn.Module):
    def __init__(self, img_info):
        super().__init__()
        H, W, C = img_info['H'], img_info['W'], img_info['C']
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Flatten(1, -1),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(0.01),
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)


class DC_G(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.model = nn.Sequential(
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
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class DC_D(nn.Module):
    def __init__(self, out_dim, img_info):
        super().__init__()
        H, W, C = img_info['H'], img_info['W'], img_info['C']
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Flatten(1, -1),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(0.01),
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(128, self.out_dim)

    def forward(self, x):
        return self.fc(self.encoder(x))


class DC_CG(nn.Module):
    def __init__(self, num_classes, noise_dim=NOISE_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim+num_classes, 1024),
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
            nn.Tanh()
        )
        self.emb = nn.Embedding(num_classes, num_classes)

    def forward(self, z, labels):
        # z = z.view(z.size(0), 100)
        c = self.emb(torch.tensor(labels).to(DEVICE))
        x = torch.cat([z, c], 1)
        # ic(x.shape)
        out = self.model(x)
        return out


if __name__ == '__main__':
    # SANITY CHECK
    pass
