import h5py
import torch
import torchvision
import numpy as np

import datasets


def convertToHDF5(path, trainset, testset, classes):
    h5file = h5py.File(path, mode='w')  # open file

    # write training data to file
    batch_size = len(trainset.data)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    _, [trainData, trainLabels] = next(enumerate(trainLoader))

    h5file.create_dataset("train/images", data=trainData)
    h5file.create_dataset("train/labels", data=trainLabels)

    # write test data to file
    batch_size = len(testset.data)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    _, [testData, testLabels] = next(enumerate(testLoader))
    h5file.create_dataset("test/images", data=testData)
    h5file.create_dataset("test/labels", data=testLabels)

    # set label to class name translation as attribute
    h5file["train/labels"].attrs['classes'] = classes
    h5file["test/labels"].attrs['classes'] = classes

    h5file.flush()
    h5file.close()


def convertMNISTToHDF5(root, path, download=True):
    trainset, testset, classes = datasets.loadTorchDatasetMNIST(root, download)
    return convertToHDF5(path, trainset, testset, classes)


def convertCIFAR10ToHDF5(root, path, download=True):
    trainset, testset, classes = datasets.loadTorchDatasetCIFAR10(root, download)
    return convertToHDF5(path, trainset, testset, classes)
