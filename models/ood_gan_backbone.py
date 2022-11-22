from config import *
from models.resnet import resnet18


def get_resnet(name, weights=None):
    # TODO: (Xiaoyang) enable direct access to more versions of ResNet
    resnets = {
        "resnet18": torchvision.models.resnet18(weights=None),
        "resnet50": torchvision.models.resnet50(weights=None),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


class Discriminator(nn.Module):
    def __init__(self, cifar_pretrained=False, resnet_version='resnet18', num_channels=3, num_class=10):
        super().__init__()
        self.cifar_pretrained = cifar_pretrained
        if self.cifar_pretrained:
            self.model = resnet18(pretrained=True)
            return
        # Network parameters
        self.num_channels = num_channels
        self.num_class = num_class

        self.encoder = get_resnet(resnet_version)
        # Deal with different channels
        if self.num_channels == 1:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(
                7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Setup final projection layer
        self.n_hidden = self.encoder.fc.out_features
        self.fc_projector = nn.Linear(self.n_hidden, self.num_class)

    def forward(self, x):
        if self.cifar_pretrained:
            return self.model(x)
        out = self.encoder(x)
        out = self.fc_projector(out)
        return out


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # Following architecture is obtained from:
        # https://learnopencv.com/deep-convolutional-gan-in-pytorch-and-tensorflow/#pytorch

        self.main = nn.Sequential(
            # Block 1:input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # Block 2: input is (64 * 8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # Block 3: input is (64 * 4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # Block 4: input is (64 * 2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Block 5: input is (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.AvgPool2d(2),
            nn.Tanh()
            # Output: output is (3) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output


if __name__ == '__main__':
    ic("OoD GAN architecture")
    D = Discriminator(cifar_pretrained=True)
    # ic(D.encoder)
    # ic(D.fc_projector)
    x = torch.zeros((10, 3,32,32))
    ic(D(x).shape)

    G = Generator(96)
    ic(G.main)
    z = torch.zeros((10, 96, 1, 1))
    ic(G(z).shape)
