from mpi4py import MPI
from abc import ABC, abstractmethod
import math
import numpy as np
import progressbar
import time
import torch
import os.path
import sys
from enum import Enum


comm = MPI.COMM_WORLD


class TrainingType(Enum):
    Sequential = 1
    AllReduce = 2


def create_training(training_type, net, criterion, optimizer, train_loader, test_loader,
                    max_epochs=math.inf, max_epochs_without_improvement=10, print_progress=False):
    """
    Creates and returns a training object of the given training type and with
    the specified parameters.
    """
    if training_type is TrainingType.Sequential:
        return SequentialTraining(net, criterion, optimizer, train_loader, test_loader, max_epochs,
                                  max_epochs_without_improvement, print_progress)
    elif training_type is TrainingType.AllReduce:
        return AllReduceTraining(net, criterion, optimizer, train_loader, test_loader, max_epochs,
                                 max_epochs_without_improvement, print_progress)
    print("Error: invalid training type")
    sys.exit(1)


def compute_avg_loss(net, dataloader, loss_criterion):
    """
    Computes and returns the average loss of the given model on the images in
    the given dataloader using the given loss_criterion.
    """
    summedLoss = 0.
    with torch.no_grad():
        net.eval()
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            loss = loss_criterion(outputs, labels)
            summedLoss += loss.item()
    return summedLoss / len(dataloader)


def compute_avg_accuracy(net, dataloader):
    """
    Computes and returns the average accuracy of the given model on the images
    in  the given dataloader by dividing the number of correctly classified
    images to the total amount of images.
    """
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        net.eval()
        for data in dataloader:
            images, labels = data
            _, predicted = torch.max(net(images), 1)
            correct = (predicted == labels).squeeze()
            correct_count += correct.sum().item()
            total_count += len(labels)
    return correct_count / total_count


def torch_dtype_to_MPI_type(dtype):
    """Converts a torch.dtype to an MPI type"""
    if dtype == torch.float or dtype == torch.half:
        return MPI.FLOAT
    elif dtype == torch.double:
        return MPI.DOUBLE
    elif dtype == torch.int64:
        return MPI.LONG
    else:
        return MPI.INT


def convert_tensor_to_MPI_buffer(tensor):
    """Converts a torch tensor to an MPI buffer"""
    pointer = tensor.data_ptr() + tensor.storage_offset()
    buffer = MPI.memory.fromaddress(pointer, 0)
    size = tensor.numel()
    mpi_type = torch_dtype_to_MPI_type(tensor.dtype)
    return [buffer, size, mpi_type]


class Training(ABC):
    """
    Abstract training class.
    Collects meta data during the training like training times and quality after each epoch.
    Start training by calling train(). Retrieve results with save_results().
    """

    header_epoch_data = "epoch\ttotalTime\tcomputationTime\tcommunicationTime\tvalidationError\tvalidationLoss\ttrainingLoss"
    header_summary_data = "epochCount\ttotalTime\tvalidationError\tvalidationLoss\tnodeCount"
    epoch_data_keys = {"epoch": 0, "totalTime": 1, "computationTime": 2,
                       "communicationTime": 3, "validationError": 4, "validationLoss": 5, "trainingLoss": 6}

    def __init__(self, net, criterion, optimizer, trainloader, testloader, max_epochs=math.inf,
                 max_epochs_without_improvement=10, print_progress=False, comm=MPI.COMM_WORLD):
        super(Training, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader

        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.min_loss = math.inf
        self.epochs_since_last_improvement = 0
        self.max_epochs_without_improvement = max_epochs_without_improvement

        self.batch_size = trainloader.batch_size
        self.comm = comm
        self.node_count = 1 if comm == None else comm.Get_size()

        self.batch_count = len(trainloader) / self.node_count
        self.epoch_data = np.zeros((1, 7))

        self.print_progress = print_progress
        if self.print_progress:
            widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
            self.epoch_progressbar = progressbar.ProgressBar(maxval=len(trainloader),
                                                             widgets=widgets)

    def save_results(self, path, comment='', config=None):
        # save epoch data
        success = True
        if os.path.exists(path + "__epochs"):
            success = False
            print("Error: cannot save epoch data at {}, file exists.".format(path + "__epochs"))
        else:
            np.savetxt(path + "__epochs", self.epoch_data, delimiter='\t',
                       comments='', header=self.header_epoch_data, footer=comment)

        # save summary data + config if given
        if os.path.exists(path + "__summary"):
            success = False
            print("Error: cannot save summary data at {}, file exists.".format(path + "__summary"))
        else:
            header = self.header_summary_data
            data = '\t'.join(map(str, self.summary_data))
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

    def add_metadata(self, key, value, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        if key in self.epoch_data_keys:
            index = self.epoch_data_keys[key]
            self.epoch_data[epoch][index] = value

    def validation_error(self):
        return 1 - compute_avg_accuracy(self.net, self.testloader)

    def validation_loss(self):
        return compute_avg_loss(self.net, self.testloader, self.criterion)

    @abstractmethod
    def epoch(self):
        pass  # must be implemented in non-abstract sub classes

    def is_root(self):
        return True

    def stop_criterion(self):
        max_epochs_criterion = self.current_epoch >= self.max_epochs
        improvement_criterion = self.epochs_since_last_improvement >= self.max_epochs_without_improvement
        return max_epochs_criterion or improvement_criterion

    def _epoch_helper(self):
        if self.is_root():
            self.epoch_data = np.append(self.epoch_data, np.zeros((1, 7)), axis=0)
            if self.print_progress:
                self.epoch_progressbar.start()
            start = time.time()

        self.epoch()

        if self.is_root():
            end = time.time()
            if self.print_progress:
                self.epoch_progressbar.finish()
                self.epoch_progressbar.finished = False

            total_time = end - start
            error = self.validation_error()
            loss = self.validation_loss()

            if loss < self.min_loss:
                self.min_loss = loss
                self.epochs_since_last_improvement = 0
            else:
                self.epochs_since_last_improvement += 1

            self.add_metadata("totalTime", total_time)
            self.add_metadata("validationError", error)
            self.add_metadata("validationLoss", loss)
            self.add_metadata("epoch", self.current_epoch + 1)

            if self.print_progress:
                msg = '\nEpochs {}\t: Running time {:.2f} s\t Error: {:.2f}%\t Validation Loss: {:.3f}\t Epochs since last improvement: {}'
                print(msg.format(self.current_epoch, total_time, error * 100, loss, self.epochs_since_last_improvement))

        if self.comm is not None:
            self.epochs_since_last_improvement = self.comm.bcast(self.epochs_since_last_improvement, root=0)
        self.current_epoch += 1

    def train(self):
        if self.is_root():
            self.add_metadata("validationError", self.validation_error(), 0)
            self.add_metadata("validationLoss", self.validation_loss(), 0)
            self.add_metadata("epoch", 0, 0)

        start = time.time()
        # loop over the training dataset until the stopCriterion is met
        while not self.stop_criterion():
            self._epoch_helper()
        if self.comm is not None:
            self.comm.Barrier()
        end = time.time()

        if self.is_root():
            self.total_time = end - start
            self.total_epochs = self.current_epoch
            self.final_validation_error = self.validation_error()
            self.final_validation_loss = self.validation_loss()
            self.summary_data = np.array([self.total_epochs, self.total_time, self.final_validation_error,
                                          self.final_validation_loss, self.node_count])


class SequentialTraining(Training):
    """Subclass of Training for simple, sequential training."""

    def __init__(self, net, criterion, optimizer, trainloader, testloader, max_epochs=math.inf,
                 max_epochs_without_improvement=10, print_progress=False):
        super(SequentialTraining, self).__init__(net, criterion, optimizer, trainloader,
                                                 testloader, max_epochs, max_epochs_without_improvement, print_progress, None)

    def epoch(self):
        self.net.train()
        summed_loss = 0.0
        for i, data in enumerate(self.trainloader, 0):
            # get inputs and zero parameter gradients
            inputs, labels = data
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()  # compute the gradients wrt to loss
            self.optimizer.step()  # use the gradients to update the model

            summed_loss += loss.item()
            if self.print_progress:
                self.epoch_progressbar.update(i + 1)
        self.add_metadata("trainingLoss", summed_loss / self.batch_count)


class AllReduceTraining(Training):
    """Subclass of Training for parallel training using MPI all-reduce."""

    def is_root(self):
        return self.comm.Get_rank() == 0

    def epoch(self):
        self.net.train()
        stats = np.zeros(3)  # loss, computation time, communication time
        reduced_stats = np.zeros(3)
        for i, data in enumerate(self.trainloader, 0):
            start_computation = time.time()
            # get inputs and zero parameter gradients
            inputs, labels = data
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()  # compute the gradients wrt to loss

            # --- naive communication
            start_communication = time.time()
            # average gradients, reduce each gradient separately
            for param in self.net.parameters():
                is_cuda = param.grad.is_cuda
                device = param.grad.device
                gradient = param.grad.cpu()

                gradient_buffer = convert_tensor_to_MPI_buffer(gradient)
                self.comm.Allreduce(MPI.IN_PLACE, gradient_buffer, op=MPI.SUM)
                gradient = gradient / self.node_count

                if is_cuda:
                    param.grad = gradient.cuda(device)
            end_communication = time.time()
            # --- actual naive communication

            self.optimizer.step()  # use the gradients to update the model
            end_computation = time.time()

            # --- statistics
            stats[0] += loss.item()
            commTime = end_communication - start_communication
            stats[1] += end_computation - start_computation - commTime
            stats[2] += commTime

            if self.is_root() and self.print_progress:
                self.epoch_progressbar.update(i + 1)

        # --- statistics
        self.comm.Reduce(stats, reduced_stats, op=MPI.SUM, root=0)
        if self.is_root():
            self.add_metadata("trainingLoss", reduced_stats[0] / self.batch_count)  # average loss per batch
            self.add_metadata("computationTime", reduced_stats[1] / self.node_count)  # average time per node
            self.add_metadata("communicationTime", reduced_stats[2] / self.node_count)
