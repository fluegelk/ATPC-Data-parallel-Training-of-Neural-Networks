import torch
import torchvision
from enum import Enum
import training
import testing
import models


class Model(Enum):
    PyTorchTutorialNet = 1
    LeNet5 = 2
    LeNet5Updated = 3


class PaddingWrapper(object):
    """
    Wrapper class for torchvision.transforms.functional.pad
    Allows passing the pad function to torchvision.transforms.Compose
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return torchvision.transforms.functional.pad(img, padding=self.padding, fill=self.fill,
                                                     padding_mode=self.padding_mode)


def loadMNIST(root, download, batch_size, num_workers):
    # pad MNIST 28x28 images with 2 px to reach same size as CIFAR (32x32)
    transform = torchvision.transforms.Compose([
        PaddingWrapper(padding=2, padding_mode='edge'),
        torchvision.transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return (trainLoader, testLoader, classes)


def loadCIFAR10(root, download, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainLoader, testLoader, classes)


def createNet(model, in_channels=3, num_classes=10):
    if model == Model.PyTorchTutorialNet:
        return models.PyTorchTutorialNet(in_channels, num_classes)
    elif model == Model.LeNet5:
        return models.LeNet5(in_channels, num_classes, updated=False)
    elif model == Model.LeNet5Updated:
        return models.LeNet5(in_channels, num_classes, updated=True)


def train(net, dataset, epochs, learning_rate=0.001, momentum=0.9):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    trainLoader, testLoader, classes = dataset

    return training.train(net, criterion, optimizer, epochs, trainLoader,
                          lambda net: 1 - testing.testAccuracy(net, testLoader, len(classes))[0])
