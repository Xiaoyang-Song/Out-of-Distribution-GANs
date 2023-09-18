from config import *
from models.resnet import resnet18
# from models.model import *
from tqdm import tqdm
from torchinfo import summary


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
    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # Following architecture is obtained from:
        # https://learnopencv.com/deep-convolutional-gan-in-pytorch-and-tensorflow/#pytorch
        if self.num_channels == 3:
            self.main = nn.Sequential(
                # Block 1:input is Z, going into a convolution
                nn.ConvTranspose2d(self.latent_dim, 64 * \
                                   8, 4, 1, 0, bias=False),
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
        elif self.num_channels == 1:
            self.main = nn.Sequential(
                # Block 1:input is Z, going into a convolution
                nn.ConvTranspose2d(self.latent_dim, 64 * \
                                   8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                # Block 2: input is (64 * 8) x 4 x 4
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                # Block 3: input is (64 * 4) x 8 x 8
                nn.ConvTranspose2d(64 * 4, 64 * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                # Block 4: input is (64 * 2) x 16 x 16
                nn.ConvTranspose2d(64 * 2, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # Block 5: input is (64) x 32 x 32
                nn.ConvTranspose2d(64, 1, 3, 2, 1, bias=False),
                nn.AvgPool2d(2),
                nn.Tanh()
                # Output: output is (3) x 32 x 32
            )

    def forward(self, input):
        output = self.main(input)
        return output


def test_backbone_D(model, val_loader):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, total_acc = [], [], []
        for idx, (img, label) in enumerate(tqdm(val_loader)):
            img, label = img.to(DEVICE), label.to(DEVICE)
            logits = model(img)
            loss = criterion(logits, label)
            num_correct, num_total = (torch.argmax(logits, dim=1) ==
                                      label).sum().item(), label.shape[0]
            acc = num_correct / num_total
            # print(acc)
            val_acc.append(acc)
            total_acc.append([num_correct, num_total])
            val_loss.append(loss.detach().item())
        print(f"Validation Accuracy: {np.mean(val_acc)}")
        print(f"Validation Classification Loss: {np.mean(val_loss)}")
        # total_acc = np.array(total_acc)
        # print(np.sum(total_acc[:, 0] / np.sum(total_acc[:, 1])))


if __name__ == '__main__':
    model = Generator(96, 3)
    noise = torch.ones(10, 96, 1, 1)
    out = model(noise)
    ic(out.shape)
    # print(summary(model))
    # print(model)
    summary(model, noise.shape)

    # from dataset import *
    # model = DC_D(8, {'H': 28, 'W': 28, 'C': 1})
    # model.load_state_dict(torch.load(
    #     "model-[64]-[15]-[2].pt", map_location=torch.device('cpu')))
    # dset = 'FashionMNIST'
    # dset = DSET(dset, True, 512, 64, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    # val_ldr = dset.ind_val_loader
    # test_backbone_D(model, val_ldr)
