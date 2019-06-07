# Train a given model on a given data set
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


def trainingEpochSequentiel(net, criterion, optimizer, trainloader):
    net.train()
    summedLoss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get inputs and zero parameter gradients
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # compute the gradients wrt to loss
        optimizer.step()  # use the gradients to update the model

        # print statistics
        summedLoss += loss.item()
    return summedLoss / len(trainloader)


def train(net, criterion, optimizer, epochs, trainloader, validationError):
    """
    Train a neural network using a given criterion and optimizer for a number of epochs on a set of training data.

    @param      net              The neural network to train
    @param      criterion        The criterion to use for the loss
    @param      optimizer        The optimizer to use
    @param      epochs           The number of epochs to train
    @param      trainloader      The data loader for the training data set
    @param      validationError  The function to compute the validation error

    @return     A list of the following meta data per epoch: training time, validation error, and average loss.
    """

    trainingTimePerEpoch = list()
    validationErrorPerEpoch = list()
    averageLossPerEpoch = list()

    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        averageLoss = trainingEpochSequentiel(net, criterion, optimizer, trainloader)
        end = time.time()

        trainingTimePerEpoch.append(end - start)
        error = validationError(net)
        validationErrorPerEpoch.append(error)
        averageLossPerEpoch.append(averageLoss)

        print('Epoch {}\t: Running time {:.2f} s\t Error: {:.1f}%'.format(epoch, end - start, error * 100))

    return trainingTimePerEpoch, validationErrorPerEpoch, averageLossPerEpoch
