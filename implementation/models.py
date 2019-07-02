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


class LeNet5(nn.Module):
    """
    Neural network LeNet-5 as defined in https://engmrk.com/lenet-5-a-classic-cnn-architecture/
    Expects inputs of size in_channelsx32x32 and gives outputs of size num_classes.
    Original LeNet-5 used avg pooling and tanh.
    """

    def __init__(self, in_channels=3, num_classes=10, updated=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)

        if updated:
            self.pool = nn.MaxPool2d(2, 2)
            self.activation = F.relu
        else:
            self.pool = nn.AvgPool2d(2, 2)
            self.activation = nn.Tanh()

        self.fc1 = nn.Linear(120 * 1 * 1, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = x.view(-1, 120 * 1 * 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class AlexNet(nn.Module):
    """
    Neural network AlexNet adjusted from the AlexNet defined in torchvision
    (see https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py).
    Changes:
    - fix position of dropout in classifier (ref. https://github.com/pytorch/vision/issues/549)
    - add variable number of input channels (for MNIST dataset)
    - remove all or last max pooling layer in features to avoid reducing the inputs to size 0,
    switch variant with noFeaturePooling parameter
    """

    def __init__(self, in_channels=3, num_classes=1000, noFeaturePooling=True):
        super(AlexNet, self).__init__()

        if noFeaturePooling:  # deactivate all max pooling layers
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:  # deactivate only the last max pooling layer
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
