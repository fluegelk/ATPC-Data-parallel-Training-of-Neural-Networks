import torch
from enum import Enum
import numpy as np
import random
import datetime
from mpi4py import MPI
import sys
import getopt
import uuid

comm = MPI.COMM_WORLD

import training
import models
import datasets


class Device(Enum):
    CPU = 1
    GPU = 2


def train(datasetType, dataset_path, modelType, dataLoaderType, trainingType,
          max_epochs, max_epochs_without_improvement, learning_rate, momentum,
          batch_size, device, printProgress=False):
    # Load Data
    path = dataset_path + datasetType.name + ".hdf5"
    trainLoader, testLoader, classes = datasets.create_dataloaders(
        dataLoaderType, path, batch_size, device, dataset=datasetType)

    # Create net and move to correct device
    in_channels = 3 if datasetType is datasets.DataSet.CIFAR10 else 1
    num_classes = 10
    net = models.create_net(modelType, in_channels, num_classes)
    net.to(device)

    # Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                momentum=momentum)

    trainObj = training.create_training(trainingType, net, criterion, optimizer, trainLoader,
                                        testLoader, max_epochs, max_epochs_without_improvement, printProgress)
    trainObj.train()
    return trainObj


def set_random_seeds(seed=0):
    random.seed(a=seed)  # set python seed
    torch.manual_seed(seed)  # set torch seed
    np.random.seed(seed)  # set numpy seed


def main(argv):
    set_random_seeds()
    torch.set_num_threads(1)

    dataset_path = '../datasets/'

    # Default Parameters
    datasetType = datasets.DataSet.MNIST
    modelType = models.Model.LeNet5Updated
    dataLoaderType = datasets.DataLoader.ParallelHDF5
    trainingType = training.TrainingType.AllReduce
    deviceType = Device.GPU

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
                 "earlystopping=", "learningrate=", "momentum=", "bs=", "batchsize=", "discard-results",
                 "print-progress", "device="]

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
                            SequentialHDF5, ParallelHDF5 [default:ParallelHDF5]

    -t --training           Select the training, values:
                            Sequential, AllReduce [default:AllReduce]

    -b --bs --batchsize     Total batch size (over all processes) [default:64]

    -e --epochs             Maximum number of training epochs [default:200]

    --device                Switch between CPU and GPU, values: CPU, GPU [default: GPU]

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
            datasetType = datasets.DataSet[arg]
        elif opt in ("-m", "--model", "--net"):
            modelType = models.Model[arg]
        elif opt in ("-l", "--dl", "--dataloader"):
            dataLoaderType = datasets.DataLoader[arg]
        elif opt in ("-t", "--training"):
            trainingType = training.TrainingType[arg]
        elif opt in ("-b", "--bs", "--batchsize"):
            batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            max_epochs = int(arg)
        elif opt in ("--device"):
            deviceType = Device[arg]
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

    if trainingType == training.TrainingType.Sequential and comm.Get_rank() != 0:
        return

    node_count = 1 if trainingType == training.TrainingType.Sequential else comm.Get_size()

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
        "device": deviceType.name}
    metadata = "\n# " + str(config)

    if comm.Get_rank() == 0:
        print('Configuration:')
        for opt, value in config.items():
            print("  {:<31}  {}".format(opt + ':', value))
        print('')

    # Assign each process a device
    if deviceType == Device.GPU and node_count > torch.cuda.device_count():
        msg = "Error: selected GPU as device but number of processes ({}) " \
            + "is greater than number of visible GPUs ({}). [Rank {}]"
        print(msg.format(node_count, torch.cuda.device_count(),
                         comm.Get_rank()))
        sys.exit(1)

    cpu_device = torch.device('cpu')
    gpu_device = torch.device('cuda', comm.Get_rank())
    device = cpu_device if deviceType == Device.CPU else gpu_device

    trainObj = train(datasetType, dataset_path, modelType, dataLoaderType,
                     trainingType, max_epochs, max_epochs_without_improvement,
                     learning_rate, momentum, batch_size, device,
                     printProgress)

    if keepResults and comm.Get_rank() == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        config["date"] = now
        unique_id = uuid.uuid4()
        trainObj.save_results("outputs/results__" + now + "__" +
                              str(unique_id), comment=metadata, config=config)


if __name__ == "__main__":
    main(sys.argv[1:])
