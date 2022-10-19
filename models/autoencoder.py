from config import *


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        # Credit: This architecture is obtained from Xiaoyang's previous research project
        super(Encoder, self).__init__()
        self.d = hidden_dim
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 x 28 x28 -> 32 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # 32 x 24 x 24 -> 32 x 12 x 12
        self.conv2 = nn.Conv2d(32, 64, 5)   # 32 x 12 x 12 -> 64 x 8 x 8
        self.pool = nn.MaxPool2d(2, 2)  # 64 x 8 x 8 -> 64 x 4 x 4
        # 64 * 4 * 4 -> 512 units dense layer
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, hidden_dim)  # 512 -> 128

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        self.d = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(self.d, 1024),
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


class AutoEncoder(nn.Module):
    def __init__(self, hidden_dim):
        self.d = hidden_dim
        self.encoder = Encoder(self.d)
        self.decoder = Decoder(self.d)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    ic('Hello autoencoder.py')
