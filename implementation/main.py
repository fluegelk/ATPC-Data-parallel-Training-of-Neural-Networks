import torch
import torchvision
from enum import Enum
import numpy as np
import random
import datetime
import math
from mpi4py import MPI
import sys
import getopt

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
    AlexNet = 4
    AlexNetPool = 5


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
    elif model is Model.AlexNet:
        return models.AlexNet(in_channels, num_classes, noFeaturePooling=True)
    elif model is Model.AlexNetPool:
        return models.AlexNet(in_channels, num_classes, noFeaturePooling=False)
    print("Error: invalid model type")


def createDataLoaders(dlType, path, batch_size, device, dataset=DataSet.CIFAR10):
    if dlType is DataLoader.SequentialHDF5:
        trainLoader = h5.HDF5DataLoader(path, batch_size, train=True, device=device)
        testLoader = h5.HDF5DataLoader(path, batch_size, train=False, shuffle=False, device=device)
        return trainLoader, testLoader, testLoader.classes
    elif dlType is DataLoader.ParallelHDF5:
        trainLoader = h5.ParallelHDF5DataLoader(path, batch_size, train=True, device=device)
        testLoader = h5.HDF5DataLoader(path, batch_size, train=False, shuffle=False, device=device)
        return trainLoader, testLoader, testLoader.classes
    elif dlType is DataLoader.Torch:
        if dataset is DataSet.CIFAR10:
            return loadTorchCIFAR10(path, True, batch_size)
        elif dataset is DataSet.MNIST:
            return loadTorchMNIST(path, True, batch_size)
    print("Error: invalid data loader type")


def createTraining(trainingType, net, criterion, optimizer, trainLoader, testLoader, max_epochs=math.inf,
                   max_epochs_without_improvement=10, printProgress=False):
    if trainingType is Training.Sequential:
        return training.SequentialTraining(net, criterion, optimizer, trainLoader,
                                           testLoader, max_epochs, max_epochs_without_improvement, printProgress)
    elif trainingType is Training.AllReduce:
        return training.AllReduceTraining(net, criterion, optimizer, trainLoader,
                                          testLoader, max_epochs, max_epochs_without_improvement, printProgress)
    print("Error: invalid training type")


def train(datasetType, dataset_path, modelType, dataLoaderType, trainingType, max_epochs,
          max_epochs_without_improvement, learning_rate, momentum, batch_size, printProgress=False):
    # Assign each process a device
    device = torch.device('cpu')  # default: cpu device
    # if rank < GPU count use GPU with id 'rank'
    if comm.Get_rank() < torch.cuda.device_count():
        device = torch.device('cuda', comm.Get_rank())

    # Load Data
    path = dataset_path if dataLoaderType is DataLoader.Torch else (dataset_path + datasetType.name + ".hdf5")
    trainLoader, testLoader, classes = createDataLoaders(dataLoaderType, path,
                                                         batch_size, device, dataset=datasetType)

    # Create net and move to correct device
    in_channels = 3 if datasetType is DataSet.CIFAR10 else 1
    num_classes = 10
    net = createNet(modelType, in_channels, num_classes)
    net.to(device)

    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    trainObj = createTraining(trainingType, net, criterion, optimizer, trainLoader, testLoader, max_epochs,
                              max_epochs_without_improvement, printProgress)
    trainObj.train()
    return trainObj


def setRandomSeeds(seed=0):
    random.seed(a=seed)  # set python seed
    torch.manual_seed(seed)  # set torch seed
    np.random.seed(seed)  # set numpy seed


def main(argv):
    setRandomSeeds()
    torch.set_num_threads(1)

    dataset_path = '../datasets/'
    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    # Default Parameters
    datasetType = DataSet.MNIST
    modelType = Model.LeNet5Updated
    dataLoaderType = DataLoader.ParallelHDF5
    trainingType = Training.AllReduce

    max_epochs = 200
    max_epochs_without_improvement = 10
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 64

    keepResults = True
    printProgress = False

    # Define command line options and help text
    short_opts = "hd:m:l:t:e:b:"
    long_opts = ["help", "data=", "model=", "net=", "dl=", "dataloader=", "training=", "epochs=",
                 "earlystopping=", "learningrate=", "momentum=", "bs=", "batchsize=", "discard-results", "print-progress"]

    help_text = """Train a neural network on an image classification problem (MNIST or CIFAR10) and collect
running times, errors and losses over time. Results are stored in 'outputs/'.
Expects the selected dataset as HDF5 file in '../datasets/'. Use createDatasets.py to create those input files.


Options:
    -h --help               Show this screen.

    -d --data               Select data set to train on
                            values: MNIST, CIFAR10 [default:MNIST]

    -m --model --net        Select model to train, values:
                            PyTorchTutorialNet, LeNet5, LeNet5Updated, AlexNet, AlexNetPool
                            [default:LeNet5Updated]

    -l --dl --dataloader    Select the data loader, values:
                            SequentialHDF5, ParallelHDF5, Torch [default:ParallelHDF5]

    -t --training           Select the training, values:
                            Sequential, AllReduce [default:AllReduce]

    -b --bs --batchsize     Total batch size (over all processes) [default:64]

    -e --epochs             Maximum number of training epochs [default:200]

    --earlystopping         Maximum number of epochs without improvement
                            before the training is stopped [default:10]

    --learningrate          Learning rate [default:0.001]

    --momentum              Training momentum [default:0.9]

    --discard-results       Do not store the training results in output/

    --print-progress        Print a progress bar and a short summary for each training epoch"""

    # Parse command line arguments
    try:
        opts, args = getopt.getopt(argv, short_opts, long_opts)
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_text)
            sys.exit()
        elif opt in ("-d", "--data"):
            datasetType = DataSet[arg]
        elif opt in ("-m", "--model", "--net"):
            modelType = Model[arg]
        elif opt in ("-l", "--dl", "--dataloader"):
            dataLoaderType = DataLoader[arg]
        elif opt in ("-t", "--training"):
            trainingType = Training[arg]
        elif opt in ("-b", "--bs", "--batchsize"):
            batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            max_epochs = int(arg)
        elif opt in ("--earlystopping"):
            max_epochs_without_improvement = int(arg)
        elif opt in ("--learningrate"):
            learning_rate = float(arg)
        elif opt in ("--momentum"):
            momentum = float(arg)
        elif opt in ("--discard-results"):
            keepResults = False
        elif opt in ("--print-progress"):
            printProgress = True

    if trainingType == Training.Sequential and comm.Get_rank() != 0:
        return

    node_count = 1 if trainingType == Training.Sequential else comm.Get_size()

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
        "node_count": node_count,
        "date": now}
    metadata = "\n# " + str(config)

    if comm.Get_rank() == 0:
        print('Configuration:')
        for opt, value in config.items():
            print("  {:<31}  {}".format(opt + ':', value))
        print('')

    trainObj = train(datasetType, dataset_path, modelType, dataLoaderType, trainingType, max_epochs,
                     max_epochs_without_improvement, learning_rate, momentum, batch_size, printProgress)

    if keepResults and comm.Get_rank() == 0:
        trainObj.saveResults("outputs/results__" + now, comment=metadata, config=config)


if __name__ == "__main__":
    main(sys.argv[1:])
