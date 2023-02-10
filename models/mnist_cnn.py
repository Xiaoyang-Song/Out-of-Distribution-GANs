from config import *


class MNISTCNN(nn.Module):
    # TODO: Modify __init__ to make it more generic later
    def __init__(self):
        # Credit: This architecture is obtained from Xiaoyang's previous research project
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 x 28 x28 -> 32 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # 32 x 24 x 24 -> 32 x 12 x 12
        self.conv2 = nn.Conv2d(32, 64, 5)   # 32 x 12 x 12 -> 64 x 8 x 8
        self.pool = nn.MaxPool2d(2, 2)  # 64 x 8 x 8 -> 64 x 4 x 4
        # 64 * 4 * 4 -> 512 units dense layer
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 512 -> 10

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MNISTCNN_AUX(nn.Module):
    # TODO: Modify __init__ to make it more generic later
    def __init__(self, C):
        self.C = C
        # Credit: This architecture is obtained from Xiaoyang's previous research project
        super(MNISTCNN_AUX, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 x 28 x28 -> 32 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # 32 x 24 x 24 -> 32 x 12 x 12
        self.conv2 = nn.Conv2d(32, 64, 5)   # 32 x 12 x 12 -> 64 x 8 x 8
        self.pool = nn.MaxPool2d(2, 2)  # 64 x 8 x 8 -> 64 x 4 x 4
        # 64 * 4 * 4 -> 512 units dense layer
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, self.C+1)  # 512 -> 10

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
