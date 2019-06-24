import torch
import torchvision
from enum import Enum
import numpy as np
import random
import datetime
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD

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


class DataLoader(Enum):
    SequentialHDF5 = 1
    ParallelHDF5 = 2
    Torch = 3


class Training(Enum):
    Sequential = 1
    AllReduce = 2


def createNet(model, in_channels=3, num_classes=10):
    if model is Model.PyTorchTutorialNet:
        return models.PyTorchTutorialNet(in_channels, num_classes)
    elif model is Model.LeNet5:
        return models.LeNet5(in_channels, num_classes, updated=False)
    elif model is Model.LeNet5Updated:
        return models.LeNet5(in_channels, num_classes, updated=True)
    print("Error: invalid model type")


def createDataLoaders(dlType, path, batch_size, dataset=DataSet.CIFAR10):
    if dlType is DataLoader.SequentialHDF5:
        trainLoader = h5.HDF5DataLoader(path, batch_size, train=True)
        testLoader = h5.HDF5DataLoader(path, batch_size, train=False, shuffle=False)
        return trainLoader, testLoader, testLoader.classes
    elif dlType is DataLoader.ParallelHDF5:
        trainLoader = h5.ParallelHDF5DataLoader(path, batch_size, train=True)
        testLoader = h5.HDF5DataLoader(path, batch_size, train=False, shuffle=False)
        return trainLoader, testLoader, testLoader.classes
    elif dlType is DataLoader.Torch:
        if dataset is DataSet.CIFAR10:
            return loadTorchCIFAR10(path, True, batch_size)
        elif dataset is DataSet.MNIST:
            return loadTorchMNIST(path, True, batch_size)
    print("Error: invalid data loader type")


def createTraining(trainingType, net, criterion, optimizer, trainLoader, testLoader, max_epochs=math.inf,
                   max_epochs_without_improvement=10):
    if trainingType is Training.Sequential:
        return training.SequentialTraining(net, criterion, optimizer, trainLoader,
                                           testLoader, max_epochs, max_epochs_without_improvement)
    elif trainingType is Training.AllReduce:
        return training.AllReduceTraining(net, criterion, optimizer, trainLoader,
                                          testLoader, max_epochs, max_epochs_without_improvement)
    print("Error: invalid training type")


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


def printDevice(net):
    param = next(net.parameters())
    if param.is_cuda:
        print("Rank {}: CUDA {}".format(comm.Get_rank(), param.get_device))
    else:
        print("Rank {}: CPU".format(comm.Get_rank()))


def main():
    setRandomSeeds()
    torch.set_num_threads(1)

    # Parameters
    datasetType = DataSet.MNIST
    modelType = Model.PyTorchTutorialNet

    dataLoaderType = DataLoader.SequentialHDF5
    trainingType = Training.Sequential

    max_epochs = 200
    max_epochs_without_improvement = 10
    learning_rate = 0.001
    momentum = 0.9

    mini_batch_size_per_node = 32
    batch_size = mini_batch_size_per_node * comm.Get_size()

    dataset_path = '../datasets/'
    in_channels = 3 if datasetType is DataSet.CIFAR10 else 1
    num_classes = 10

    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")

    config = {
        "dataset": datasetType.name,
        "dataLoader": dataLoaderType.name,
        "model": modelType.name,
        "training": trainingType.name,
        "max_epochs": max_epochs,
        "max_epochs_without_improvement": max_epochs_without_improvement,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "batch_size": batch_size,
        "date": now}
    metadata = "\n# " + str(config)

    # Load Data
    path = dataset_path if dataLoaderType is DataLoader.Torch else (dataset_path + datasetType.name + ".hdf5")
    trainLoader, testLoader, classes = createDataLoaders(dataLoaderType, path, batch_size, dataset=datasetType)

    # Training
    net = createNet(modelType, in_channels, num_classes)
    printDevice(net)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    trainObj = createTraining(trainingType, net, criterion, optimizer, trainLoader, testLoader, max_epochs,
                              max_epochs_without_improvement)
    trainObj.train()

    if comm.Get_rank() == 0:
        trainObj.saveResults("outputs/results__" + now, comment=metadata)


if __name__ == "__main__":
    main()
