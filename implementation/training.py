from abc import ABC, abstractmethod
import math
import progressbar
import time
import torch


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

    def __init__(self, net, criterion, optimizer, trainloader, testloader, max_epochs=math.inf):
        super(Training, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader

        self.current_epoch = 0
        self.max_epochs = max_epochs

        # element_count = dataloader.dataset.data.shape[0]
        # self.batch_count = math.ceil(element_count / dataloader.batch_size)
        self.batch_count = len(trainloader)

        self.epoch_progressbar = progressbar.ProgressBar(maxval=self.batch_count, widgets=[progressbar.Bar(
            '=', '[', ']'), ' ', progressbar.Percentage()])

        self.metadata = {}

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
        return self.current_epoch >= self.max_epochs

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
            self.addMetadata("trainingTime", trainingTime)
            self.addMetadata("validationError", error)
            self.addMetadata("validationLoss", loss)

            msg = 'Epoch {}\t: Running time {:.2f} s\t Error: {:.1f}%'
            print(msg.format(self.current_epoch, trainingTime, error * 100))
            self.current_epoch += 1

    def train(self):
        # loop over the training dataset until the stopCriterion is met
        while not self.stopCriterion():
            self.epochHelper()


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
