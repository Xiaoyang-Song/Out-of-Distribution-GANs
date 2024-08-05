# Credit: this code is borrowed from https://github.com/wyn430/WOOD with minor modification.
from config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.dc_gan_model import *
from models.ood_gan_backbone import *
from models.resnet import *


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0, input_channel=1):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(input_channel, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(
            math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(
            math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # for three channel images: 3 x 32 x 32
        # self.fc1 = nn.Linear(in_planes, 128)
        # self.fc2 = nn.Linear(128, num_classes)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # ic(out.shape)
        out = F.avg_pool2d(out, 7)
        out = out.view(-1, self.in_planes)
        # ic(out.shape)
        out = self.fc(out)
        # out = self.softmax(out)
        return out
class PCDiscriminator(nn.Module):
    def __init__(self):
        super(PCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding = (0, 20))
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 900)
        self.gate = nn.Tanh()

    def forward(self, x):
        out = x.squeeze()
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.gate(self.fc4(out).reshape(-1, 1, 300, 3))
        return out

class MODEL_GETTER():
    def __init__(self, num_classes, img_info, return_DG):
        self.num_classes = num_classes
        self.img_info = img_info
        self.H, self.W, self.C = self.img_info.values()
        self.return_DG = return_DG

    def __call__(self, D_model, D_config=None, G_model=None, G_config=None, device=DEVICE):
        if self.return_DG:
            assert G_model is not None
            assert G_config is not None
            assert 'noise_dim' in G_config and G_config['noise_dim'] is not None
        # Process DISCRIMINATORS
        if D_model == 'DC_D':
            D = DC_D(self.num_classes, self.img_info).to(device)
        elif D_model == 'DenseNet':
            # TODO: add depth control argument
            assert D_config is not None
            assert 'depth' in D_config and D_config['depth'] is not None
            D = DenseNet3(depth=D_config['depth'], num_classes=self.num_classes,
                          input_channel=self.C).to(device)
        elif D_model == 'ResNet':
            D = resnet101(num_class=self.num_classes).to(device)
        elif D_model == '3DPC':
            D = PCDiscriminator().to(device)
        else:
            assert False, 'Unrecognized Discriminator Type.'
        # Process GENERATORS
        if G_model is not None:
            noise_dim = G_config['noise_dim']
            if G_model == 'DC_CG':
                G = DC_CG(self.num_classes, noise_dim).to(device)
            elif G_model == 'DC_G':
                G = DC_G(noise_dim).to(device)
            elif G_model == "Deep_G":
                G = Generator(noise_dim, self.C).to(device)
            elif G_model == 'Deep_ResNet_G':
                config = [3, 4, 6, 3]
                G = ResNetDecoder(config[::-1], False).to(device)
            elif G_model == "3DPC":
                G = PCGenerator().to(device)
            else:
                assert False, 'Unrecognized Generator Type.'
        if self.return_DG:
            return D, G
        else:
            return D


if __name__ == '__main__':
    pass
    C = 3
    model = DenseNet3(depth=100, num_classes=10, input_channel=C)
    model.to(DEVICE)

    ic("OoD GAN architecture")
    model = nn.DataParallel(DenseNet3(100, 10, input_channel=3))
    out = model(torch.ones(1, 3, 32, 32))
    ic(out.shape)
    # state_dict = torch.load("other/model.t7", map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)

    # from dataset import *
    # _, _, _, val_ldr = FashionMNIST(256, 64, True)
    # test_backbone_D(model, val_ldr)
    # for Binary WOOD
    # 0.9348999999999998 for Dynamic WOOD
    # 0.8996 for Binary WOOD