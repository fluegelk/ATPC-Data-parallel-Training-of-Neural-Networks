from mpi4py import MPI
from abc import ABC, abstractmethod
import math
import numpy as np
import progressbar
import time
import torch
import os.path

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

    header_epochData = "epoch\ttotalTime\tcomputationTime\tcommunicationTime\tvalidationError\tvalidationLoss\ttrainingLoss"
    header_summaryData = "epochCount\ttotalTime\tvalidationError\tvalidationLoss\tnodeCount"
    epochDataKeys = {"epoch": 0, "totalTime": 1, "computationTime": 2,
                     "communicationTime": 3, "validationError": 4, "validationLoss": 5, "trainingLoss": 6}

    def __init__(self, net, criterion, optimizer, trainloader, testloader, max_epochs=math.inf,
                 max_epochs_without_improvement=10, printProgress=False, comm=MPI.COMM_WORLD):
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

        self.batch_size = trainloader.batch_size
        self.comm = comm
        self.node_count = 1 if comm == None else comm.Get_size()

        self.batch_count = len(trainloader) / self.node_count
        self.epochData = np.zeros((1, 7))

        self.printProgress = printProgress
        if self.printProgress:
            widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
            self.epoch_progressbar = progressbar.ProgressBar(maxval=len(trainloader),
                                                             widgets=widgets)

    def saveResults(self, path, comment='', config=None):
        # save epoch data
        success = True
        if os.path.exists(path + "__epochs"):
            success = False
            print("Error: cannot save epoch data at {}, file exists.".format(path + "__epochs"))
        else:
            np.savetxt(path + "__epochs", self.epochData, delimiter='\t',
                       comments='', header=self.header_epochData, footer=comment)

        # save summary data + config if given
        if os.path.exists(path + "__summary"):
            success = False
            print("Error: cannot save summary data at {}, file exists.".format(path + "__summary"))
        else:
            header = self.header_summaryData
            data = '\t'.join(map(str, self.summaryData))
            if config is not None:
                for opt, value in config.items():
                    header = header + '\t' + opt
                    data = data + '\t' + str(value)

            file = open(path + "__summary", "w")
            file.write(header + "\n")
            file.write(data + "\n")
            file.write(comment + "\n")
            file.close()

        if success:
            print("Training results successfully saved at " + path)
        else:
            print("Error while trying to save Training results at " + path)

    def addMetadata(self, key, value, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        if key in self.epochDataKeys:
            index = self.epochDataKeys[key]
            self.epochData[epoch][index] = value

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
            self.epochData = np.append(self.epochData, np.zeros((1, 7)), axis=0)
            if self.printProgress:
                self.epoch_progressbar.start()
            start = time.time()

        self.epoch()

        if self.is_root():
            end = time.time()
            if self.printProgress:
                self.epoch_progressbar.finish()
                self.epoch_progressbar.finished = False

            totalTime = end - start
            error = self.validationError()
            loss = self.validationLoss()

            if loss < self.bestValidationLoss:
                self.bestValidationLoss = loss
                self.epochsSinceLastImprovement = 0
            else:
                self.epochsSinceLastImprovement += 1

            self.addMetadata("totalTime", totalTime)
            self.addMetadata("validationError", error)
            self.addMetadata("validationLoss", loss)
            self.addMetadata("epoch", self.current_epoch + 1)

            if self.printProgress:
                msg = '\nEpochs {}\t: Running time {:.2f} s\t Error: {:.2f}%\t Validation Loss: {:.3f}\t Epochs since last improvement: {}'
                print(msg.format(self.current_epoch, totalTime, error * 100, loss, self.epochsSinceLastImprovement))

        if self.comm != None:
            self.epochsSinceLastImprovement = self.comm.bcast(self.epochsSinceLastImprovement, root=0)
        self.current_epoch += 1

    def train(self):
        if self.is_root():
            self.addMetadata("validationError", self.validationError(), 0)
            self.addMetadata("validationLoss", self.validationLoss(), 0)
            self.addMetadata("epoch", 0, 0)

        start = time.time()
        # loop over the training dataset until the stopCriterion is met
        while not self.stopCriterion():
            self.epochHelper()
        if self.comm != None:
            self.comm.Barrier()
        end = time.time()

        if self.is_root():
            self.totalTime = end - start
            self.totalEpochs = self.current_epoch
            self.finalValidationError = self.validationError()
            self.finalValidationLoss = self.validationLoss()
            self.summaryData = np.array([self.totalEpochs, self.totalTime, self.finalValidationError,
                                         self.finalValidationLoss, self.node_count])


class SequentialTraining(Training):
    """docstring for SequentialTraining"""

    def __init__(self, net, criterion, optimizer, trainloader, testloader, max_epochs=math.inf,
                 max_epochs_without_improvement=10, printProgress=False):
        super(SequentialTraining, self).__init__(net, criterion, optimizer, trainloader,
                                                 testloader, max_epochs, max_epochs_without_improvement, printProgress, None)

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
            if self.printProgress:
                self.epoch_progressbar.update(i + 1)
        self.addMetadata("trainingLoss", summedLoss / self.batch_count)


class AllReduceTraining(Training):
    """docstring for AllReduceTraining"""

    def is_root(self):
        return self.comm.Get_rank() == 0

    def epoch(self):
        self.net.train()
        stats = np.zeros(3)  # loss, computation time, communication time
        reducedStats = np.zeros(3)
        for i, data in enumerate(self.trainloader, 0):
            startComputation = time.time()
            # get inputs and zero parameter gradients
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
                is_cuda = param.grad.is_cuda
                device = param.grad.device
                gradient = param.grad.cpu()

                gradientBuffer = convertPyTorchTensorToMPIBuffer(gradient)
                self.comm.Allreduce(MPI.IN_PLACE, gradientBuffer, op=MPI.SUM)
                gradient = gradient / self.node_count

                if is_cuda:
                    param.grad = gradient.cuda(device)
            endCommunication = time.time()
            # --- actual naive communication

            self.optimizer.step()  # use the gradients to update the model
            endComputation = time.time()

            # --- statistics
            stats[0] += loss.item()
            commTime = endCommunication - startCommunication
            stats[1] += endComputation - startComputation - commTime
            stats[2] += commTime

            if self.is_root() and self.printProgress:
                self.epoch_progressbar.update(i + 1)

        # --- statistics
        self.comm.Reduce(stats, reducedStats, op=MPI.SUM, root=0)
        if self.is_root():
            self.addMetadata("trainingLoss", reducedStats[0] / self.batch_count)  # average loss per batch
            self.addMetadata("computationTime", reducedStats[1] / self.node_count)  # average time per node
            self.addMetadata("communicationTime", reducedStats[2] / self.node_count)
