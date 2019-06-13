from mpi4py import MPI
from abc import ABC, abstractmethod
import math
import numpy as np
import progressbar
import time
import torch

import testing

comm = MPI.COMM_WORLD


def torch_dtype_to_MPI_type(dtype):
    if dtype == torch.float or dtype == torch.half:
        return MPI.FLOAT
    elif dtype == torch.double:
        return MPI.DOUBLE
    elif dtype == torch.int64:
        return MPI.LONG
    else:
        return MPI.INT


def convertPyTorchTensorToMPIBuffer(tensor):
    pointer = tensor.data_ptr() + tensor.storage_offset()
    buffer = MPI.memory.fromaddress(pointer, 0)
    size = tensor.numel()
    mpi_type = torch_dtype_to_MPI_type(tensor.dtype)
    return [buffer, size, mpi_type]


def saveCheckpoint(path, net, optimizer, epoch):
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)


def loadCheckpoint(path, net, optimizer):
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return epoch


class Training(ABC):
    """docstring for Training"""

    def __init__(self, net, criterion, optimizer, trainloader, testloader, max_epochs=math.inf,
                 max_epochs_without_improvement=10):
        super(Training, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader

        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.bestValidationLoss = math.inf
        self.epochsSinceLastImprovement = 0
        self.max_epochs_without_improvement = max_epochs_without_improvement

        self.batch_count = len(trainloader)
        self.epoch_progressbar = progressbar.ProgressBar(maxval=self.batch_count, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])

        self.metadata = {}
        self.totalTime = 0.

    def addMetadata(self, key, value):
        if key not in self.metadata:
            self.metadata[key] = list()
        self.metadata[key].append((self.current_epoch, value))

    def validationError(self):
        return 1 - testing.computeAccuracy(self.net, self.testloader)

    def validationLoss(self):
        return testing.computeAverageLoss(self.net, self.testloader, self.criterion)

    @abstractmethod
    def epoch(self):
        pass  # must be implemented in non-abstract sub classes

    def is_root(self):
        return True

    def stopCriterion(self):
        maxEpochsCriterion = self.current_epoch >= self.max_epochs
        improvementCriterion = self.epochsSinceLastImprovement >= self.max_epochs_without_improvement
        return maxEpochsCriterion or improvementCriterion

    def epochHelper(self):
        if self.is_root():
            self.epoch_progressbar.start()
            start = time.time()

        self.epoch()

        if self.is_root():
            end = time.time()
            self.epoch_progressbar.finish()

            trainingTime = end - start
            error = self.validationError()
            loss = self.validationLoss()

            if loss < self.bestValidationLoss:
                self.bestValidationLoss = loss
                self.epochsSinceLastImprovement = 0
            else:
                self.epochsSinceLastImprovement += 1

            self.addMetadata("trainingTime", trainingTime)
            self.addMetadata("validationError", error)
            self.addMetadata("validationLoss", loss)

            msg = '\nEpoch {}\t: Running time {:.2f} s\t Error: {:.1f}%\t Validation Loss: {:.2f}'
            print(msg.format(self.current_epoch, trainingTime, error * 100, loss))
            self.current_epoch += 1

    def train(self):
        start = time.time()
        # loop over the training dataset until the stopCriterion is met
        while not self.stopCriterion():
            self.epochHelper()
        comm.Barrier()

        if self.is_root():
            end = time.time()
            self.totalTime


class SequentialTraining(Training):
    """docstring for SequentialTraining"""

    def epoch(self):
        self.net.train()
        summedLoss = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get inputs and zero parameter gradients
            inputs, labels = data
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()  # compute the gradients wrt to loss
            self.optimizer.step()  # use the gradients to update the model

            summedLoss += loss.item()
            self.epoch_progressbar.update(i + 1)
        self.addMetadata("summedLoss", summedLoss / self.batch_count)


class AllReduceTraining(Training):
    """docstring for AllReduceTraining"""

    def is_root(self):
        return comm.Get_rank() == 0

    def epoch(self):
        self.net.train()
        stats = np.zeros(3)  # loss, computation time, communication time
        for i, data in enumerate(self.trainloader, 0):
            startComputation = time.time()
            # get inputs and zero parameter gradients
            # TODO: get subset of batch relevant for this node
            inputs, labels = data
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()  # compute the gradients wrt to loss

            # --- naive communication
            startCommunication = time.time()
            # average gradients, reduce each gradient separately
            for param in self.net.parameters():
                gradientBuffer = convertPyTorchTensorToMPIBuffer(param.grad)
                comm.Allreduce(MPI.IN_PLACE, gradientBuffer, op=MPI.SUM)
                param.grad = param.grad / comm.Get_size()
            endCommunication = time.time()
            # --- actual naive communication

            self.optimizer.step()  # use the gradients to update the model
            endComputation = time.time()

            # --- statistics
            stats[0] += loss.item()
            commTime = endCommunication - startCommunication
            stats[1] += endComputation - startComputation - commTime
            stats[2] += commTime

            if self.is_root():
                self.epoch_progressbar.update(i + 1)

        # --- statistics
        comm.Reduce(MPI.IN_PLACE, stats, op=MPI.SUM, root=0)
        if self.is_root():
            self.addMetadata("summedLoss", stats[0] / self.batch_count)
            self.addMetadata("computationTime", stats[1] / self.batch_count)
            self.addMetadata("communicationTime", stats[2] / self.batch_count)
