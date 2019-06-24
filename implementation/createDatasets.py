import HDF5Adapter as h5
from datasets import *

root = "../datasets/"
download = True

trainset, testset, classes = loadTorchDatasetMNIST(root, download)
h5.convertToHDF5(root + "MNIST.hdf5", trainset, testset, classes)

trainset, testset, classes = loadTorchDatasetCIFAR10(root, download)
h5.convertToHDF5(root + "CIFAR10.hdf5", trainset, testset, classes)
