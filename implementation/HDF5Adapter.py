import h5py
import math
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


class HDF5DataLoader(object):
    """docstring for HDF5DataLoader"""

    def __init__(self, path, batch_size, train):
        self.h5file = h5py.File(path, mode='r')  # open file

        self.train = train
        self.group = "train" if train else "test"

        self.labels = self.h5file[self.group + "/labels"]
        self.images = self.h5file[self.group + "/images"]
        self.classes = self.labels.attrs['classes']

        self.sample_count = len(self.labels)
        self.set_batch_size(batch_size)
        self.currentBatch = 0

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_count = math.ceil(self.sample_count / self.batch_size)

    def _batch_start_index(self, batch_index):
        if batch_index >= self.batch_count:
            return self.sample_count
        else:
            return batch_index * self.batch_size

    def get_batch(self, batch_index):
        start_index = self._batch_start_index(batch_index)
        end_index = self._batch_start_index(batch_index + 1)
        images = torch.from_numpy(self.images[start_index:end_index])
        labels = torch.from_numpy(self.labels[start_index:end_index])
        return [images, labels]

    def __getitem__(self, index):
        return self.get_batch(index)

    def __len__(self):
        return self.batch_count

    def __next__(self):
        if self.currentBatch < self.batch_count:
            batch = self.__getitem__(self.currentBatch)
            self.currentBatch += 1
            return batch
        raise StopIteration()

    def __iter__(self):
        # TODO: allow calling __iter__ multiple times concurrently on the same data loader?
        self.currentBatch = 0
        return self
