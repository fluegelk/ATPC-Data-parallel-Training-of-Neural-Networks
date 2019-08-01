import torchvision

import sys
from enum import Enum

import HDF5Adapter as h5


class DataSet(Enum):
    MNIST = 1
    CIFAR10 = 2


class DataLoader(Enum):
    SequentialHDF5 = 1
    ParallelHDF5 = 2


def create_dataloaders(dataloader_type, path, batch_size, device, dataset=DataSet.CIFAR10):
    """
    Creates and returns two dataloaders of the type given by dataloader_type. Loads the data from the given path.
    The data is organized in batches of size batch_size and allocated on the given device.
    """
    if dataloader_type is DataLoader.SequentialHDF5:
        trainloader = h5.HDF5DataLoader(path, batch_size, train=True, device=device)
        testloader = h5.HDF5DataLoader(path, batch_size, train=False, shuffle=False, device=device)
        return trainloader, testloader, testloader.classes
    elif dataloader_type is DataLoader.ParallelHDF5:
        trainloader = h5.ParallelHDF5DataLoader(path, batch_size, train=True, device=device)
        testloader = h5.HDF5DataLoader(path, batch_size, train=False, shuffle=False, device=device)
        return trainloader, testloader, testloader.classes
    print("Error: invalid data loader type")
    sys.exit(1)


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


def load_torchvision_MNIST(root, download):
    """
    Loads torchvision.datasets.MNIST, downloads required files if necessary.
    Returns a tuple of the training set, the test set and the class names.
    Images are padded with 0 to size 32x32 and converted to tensor.
    """
    # pad MNIST 28x28 images with 2 px to reach same size as CIFAR (32x32)
    transform = torchvision.transforms.Compose([
        PaddingWrapper(padding=2, padding_mode='edge'),
        torchvision.transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return trainset, testset, classes


def load_torchvision_CIFAR10(root, download):
    """
    Loads torchvision.datasets.CIFAR10, downloads required files if necessary.
    Returns a tuple of the training set, the test set and the class names.
    Images are converted to tensor and normalized.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, testset, classes


def create_HDF5_datasets():
    """
    Downloads MNIST and CIFAR10 datasets from torchvision to ../datasets/
    Converts both datasets to HDF5 and stores them at ../datasets/{MNIST,CIFAR10}.hdf5
    """
    root = "../datasets/"
    download = True

    print("Downloading MNIST:")
    trainset, testset, classes = load_torchvision_MNIST(root, download)
    print("Converting MNIST to HDF5:")
    h5.convert_to_HDF5(root + "MNIST.hdf5", trainset, testset, classes)
    print("MNIST Done!")

    print("Downloading CIFAR10:")
    trainset, testset, classes = load_torchvision_CIFAR10(root, download)
    print("Converting CIFAR10 to HDF5:")
    h5.convert_to_HDF5(root + "CIFAR10.hdf5", trainset, testset, classes)
    print("CIFAR10 Done!")

if __name__ == "__main__":
    create_HDF5_datasets()
