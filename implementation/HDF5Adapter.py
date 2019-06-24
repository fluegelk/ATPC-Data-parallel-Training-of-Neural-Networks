import h5py
import math
import torch
import torchvision
import numpy as np
from mpi4py import MPI

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

    def __init__(self, path, batch_size, train, shuffle=True, device=torch.device('cpu')):
        self.device = device
        self.h5file = h5py.File(path, mode='r')  # open file

        self.train = train
        self.group = "train" if train else "test"

        self.labels_ref = self.h5file[self.group + "/labels"]
        self.images_ref = self.h5file[self.group + "/images"]
        self.classes = self.labels_ref.attrs['classes']

        self.sample_count = len(self.labels_ref)
        self.set_batch_size(batch_size)
        self.currentBatch = 0
        self.shuffle = shuffle

        self._load_data()
        self._to_tensor()

    def _load_data(self):
        self.labels = self.labels_ref[:]
        self.images = self.images_ref[:]

    def _to_tensor(self):
        self.images_tensor = torch.from_numpy(self.images).to(self.device)
        self.labels_tensor = torch.from_numpy(self.labels).to(self.device)

    def shuffle_data(self):
        rng_state = np.random.get_state()  # store random state
        np.random.shuffle(self.labels)
        np.random.set_state(rng_state)  # restore random state to ensure equal shuffling for both arrays
        np.random.shuffle(self.images)
        self._to_tensor()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.batch_count = math.floor(self.sample_count / self.batch_size)

    def _batch_start_index(self, batch_index):
        if batch_index >= self.batch_count:
            return self.sample_count
        else:
            return batch_index * self.batch_size

    def _batch_end_index(self, batch_index):
        return self._batch_start_index(batch_index + 1)

    def get_batch(self, batch_index):
        start_index = self._batch_start_index(batch_index)
        end_index = self._batch_end_index(batch_index)
        images = self.images_tensor[start_index:end_index]
        labels = self.labels_tensor[start_index:end_index]
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
        if self.shuffle:
            self.shuffle_data()
        raise StopIteration()

    def __iter__(self):
        # TODO: allow calling __iter__ multiple times concurrently on the same data loader?
        self.currentBatch = 0
        return self


class ParallelHDF5DataLoader(HDF5DataLoader):
    """docstring for ParallelHDF5DataLoader"""

    def __init__(self, path, batch_size, train, shuffle=True, device=torch.device('cpu'), comm=MPI.COMM_WORLD):
        self.comm = comm
        self.node_count = 1 if comm == None else comm.Get_size()
        super().__init__(path, batch_size, train, shuffle, device)

    def _load_data(self):
        node_start_index = self.comm.Get_rank() * self.sample_count_per_node
        node_end_index = (self.comm.Get_rank() + 1) * self.sample_count_per_node
        self.images = self.images_ref[node_start_index:node_end_index]
        self.labels = self.labels_ref[node_start_index:node_end_index]

    def set_batch_size(self, batch_size):
        assert (batch_size % self.node_count == 0), "Batch size must be divisible by MPI node count!"
        self.batch_size = batch_size
        self.batch_size_per_node = math.ceil(self.batch_size / self.node_count)

        self.sample_count_per_node = math.floor(self.sample_count / self.node_count)
        self.batch_count = math.floor(self.sample_count_per_node / self.batch_size)

    def _mini_batch_start_index(self, batch_index):
        if batch_index >= self.batch_count:
            return self.sample_count_per_node
        else:
            return batch_index * self.batch_size_per_node

    def _mini_batch_end_index(self, batch_index):
        return self._mini_batch_start_index(batch_index + 1)

    def get_mini_batch(self, batch_index):
        start_index = self._mini_batch_start_index(batch_index)
        end_index = self._mini_batch_end_index(batch_index)
        images = self.images_tensor[start_index:end_index]
        labels = self.labels_tensor[start_index:end_index]
        return [images, labels]

    def __getitem__(self, index):
        return self.get_mini_batch(index)
