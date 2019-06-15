import torch
import torchvision
from enum import Enum
import numpy as np
import random
import datetime

import HDF5Adapter as h5
import training
import testing
import models
import visualisation as vis
from datasets import *


class Model(Enum):
    PyTorchTutorialNet = 1
    LeNet5 = 2
    LeNet5Updated = 3


def createNet(model, in_channels=3, num_classes=10):
    if model == Model.PyTorchTutorialNet:
        return models.PyTorchTutorialNet(in_channels, num_classes)
    elif model == Model.LeNet5:
        return models.LeNet5(in_channels, num_classes, updated=False)
    elif model == Model.LeNet5Updated:
        return models.LeNet5(in_channels, num_classes, updated=True)


def printAccuracy(net, dataset):
    _, testLoader, classes = dataset
    totalAccuracy, accuracyByClass = testing.computeAccuracy(net, testLoader, len(classes))
    print('Total Accuracy: %d %%' % (100 * totalAccuracy))
    testing.printClassAccuracy(classes, accuracyByClass)


def showTestBatch(net, testLoader, classes, prepFunc=None):
    images, actual = next(iter(testLoader))
    _, predicted = torch.max(net(images), 1)
    if prepFunc is not None:
        images = list(map(prepFunc, images))
    vis.showImgagesAsGrid(images, vis.actualVsPredictedClass(actual, predicted, classes))


def setRandomSeeds(seed=0):
    random.seed(a=seed)  # set python seed
    torch.manual_seed(seed)  # set torch seed
    np.random.seed(seed)  # set numpy seed


def main():
    setRandomSeeds()
    # Parameters
    _dataset = DataSet.MNIST
    in_channels = 3 if _dataset == DataSet.CIFAR10 else 1
    num_classes = 10

    dataset_path = '../datasets/'
    download = False
    batch_size = 32
    dataloader_workers = 2

    epochs = 5
    learning_rate = 0.001
    momentum = 0.9
    model = Model.PyTorchTutorialNet

    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")

    # Load Data
    trainLoader, testLoader, classes = None, None, None
    if _dataset is DataSet.CIFAR10:
        trainLoader = h5.HDF5DataLoader(dataset_path + "CIFAR10.hdf5", batch_size, train=True)
        testLoader = h5.HDF5DataLoader(dataset_path + "CIFAR10.hdf5", batch_size, train=False)
        classes = testLoader.get_classes()
    else:
        trainLoader = h5.HDF5DataLoader(dataset_path + "MNIST.hdf5", batch_size, train=True)
        testLoader = h5.HDF5DataLoader(dataset_path + "MNIST.hdf5", batch_size, train=False)
        classes = testLoader.get_classes()

    # Training
    net = createNet(model, in_channels, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    trainObj = training.SequentialTraining(net, criterion, optimizer, trainLoader, testLoader, epochs)
    trainObj.train()
    trainObj.saveMetadata("outputs/results__" + now)


if __name__ == "__main__":
    main()
