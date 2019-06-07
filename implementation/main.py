import torch
import torchvision
from enum import Enum

import training
import testing
import models
import visualisation as vis


class Model(Enum):
    PyTorchTutorialNet = 1
    LeNet5 = 2
    LeNet5Updated = 3


class DataSet(Enum):
    MNIST = 1
    CIFAR10 = 2


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


def printAccuracy(net, dataset):
    _, testLoader, classes = dataset
    totalAccuracy, accuracyByClass = testing.testAccuracy(net, testLoader, len(classes))
    print('Total Accuracy: %d %%' % (100 * totalAccuracy))
    testing.printClassAccuracy(classes, accuracyByClass)


def showTestBatch(net, dataset, prepFunc=None):
    _, testLoader, classes = dataset
    images, actual = iter(testLoader).next()
    _, predicted = torch.max(net(images), 1)
    if prepFunc is not None:
        images = list(map(prepFunc, images))
    vis.showImgagesAsGrid(images, vis.actualVsPredictedClass(actual, predicted, classes))


def main():
    # Parameters
    _dataset = DataSet.MNIST
    in_channels = 3 if _dataset == DataSet.CIFAR10 else 1
    num_classes = 10

    dataset_path = '../datasets'
    download = False
    batch_size = 32
    dataloader_workers = 2

    epochs = 5
    _model = Model.LeNet5Updated

    # Load Data
    dataset = None
    if _dataset is DataSet.CIFAR10:
        dataset = loadCIFAR10(dataset_path, download, batch_size, dataloader_workers)
    else:
        dataset = loadMNIST(dataset_path, download, batch_size, dataloader_workers)

    # Training
    net = createNet(_model, in_channels, num_classes)
    trainingMetaData = train(net, dataset, epochs)

    # Evaluation & Visualisation
    vis.plotTrainingMetaData(trainingMetaData)
    printAccuracy(net, dataset)
    plotPrepFunc = vis.CIFARImgagePlotPreparation if _dataset == DataSet.CIFAR10 else vis.MNISTImgagePlotPreparation
    showTestBatch(net, dataset, plotPrepFunc)


if __name__ == "__main__":
    main()
