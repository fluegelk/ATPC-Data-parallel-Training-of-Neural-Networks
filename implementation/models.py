# Definition of neural networks with rewritten backwards function for data parallelism
import torch.nn as nn
import torch.nn.functional as F


class PyTorchTutorialNet(nn.Module):
    """
    Neural network as defined in this PyTorch Deep Learning Tutorial:
    https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
    Expects inputs of size in_channelsx32x32 and gives outputs of size num_classes.

    Output sizes:
        input: in_channelsx32x32
        conv1: 6x28x28
        pool1: 6x14x14
        conv2: 16x10x10
        pool2: 16x5x5
        fc1: 120
        fc2: 84
        fc3: num_classes
    """

    def __init__(self, in_channels=3, num_classes=10):
        super(PyTorchTutorialNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
