import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import torch


def CIFARImagePlotPreparation(image):
    image = image / 2 + 0.5     # unnormalize
    npimg = image.numpy()
    return np.transpose(npimg, (1, 2, 0))  # transpose (RGB channels to end)


def MNISTImagePlotPreparation(image):
    img = np.transpose(image.numpy(), (1, 2, 0))  # transpose
    return np.squeeze(img)


def labelsToClassName(labels, classes):
    return list(map(lambda i: classes[i], labels))


def actualVsPredictedClass(actualLabels, predictedLabels, classes):
    """
    Takes two lists of image labels (actual and predicted) and a list of class
    names. Returns a list actual and predicted class names separated by a new
    line.
    """
    actualClasses = labelsToClassName(actualLabels, classes)
    predictedClasses = labelsToClassName(predictedLabels, classes)

    def formatString(actual, predicted):
        return '{}\n{}'.format(actual, predicted)
    return list(map(formatString, actualClasses, predictedClasses))


def showImgagesAsGrid(images, labels, ncols=10):
    """
    Plot the given images using pyplot in a grid with ncol columns.
    The given labels are used as image captions.
    """
    nrows = -(len(images) // -ncols)  # = ceil(len(images) / ncols)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # plot images with labels
    for i in range(len(images)):
        axes.ravel()[i].imshow(images[i])
        axes.ravel()[i].set_title(labels[i])
        axes.ravel()[i].set_axis_off()

    # remove axis for empty subplots
    for i in range(len(images), nrows * ncols):
        axes.ravel()[i].set_axis_off()

    plt.show()

# Examples for image printing:
# images, labels = iter(trainLoader).next()
# vis.showImgagesAsGrid(list(map(vis.CIFARImgagePlotPreparation, images)),
#                       labelsToClassName(labels, CIFAR10_Classes))

# images, actualLabels = iter(testLoader).next()
# _, predictedLabels = torch.max(net(images), 1)
# showImgagesAsGrid(list(map(CIFARImgagePlotPreparation, images)),
#                   actualVsPredictedClass(actualLabels, predictedLabels,
#                                          CIFAR10_Classes))


def plotTrainingMetaData(trainingMetaData):
    """
    Plot a line plot of the training time, validation error and average loss
    per epoch and store them as PDF in a new directory '%now--trainingData' in
    'output'. Prints the exact output directory to console.
    """
    trainingTime, averageLoss, validationError = trainingMetaData
    epochs = len(trainingTime)

    now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    path = "outputs/" + now + "--traningData"

    if not os.path.exists(path):
        os.makedirs(path)

    def plotAndSave(data, label):
        plt.plot([i for i in range(epochs)], data, label=label)
        plt.legend(loc=1, mode='expanded', shadow=True, ncol=2)
        plt.savefig(path + '/' + label)
        plt.close()

    plotAndSave(data=trainingTime, label="trainingTime")
    plotAndSave(data=validationError, label="validationError")
    plotAndSave(data=averageLoss, label="averageLoss")

    print("Training plots stored at " + path)


def computeAccuracyByClass(net, dataloader, numClasses):
    """
    @brief      Test a given (trained) model on a set of test data.
                Determine total accuracy and accuracy per class.

    @param      net         The trained model
    @param      dataloader  The data loader for the test data
    @param      numClasses  The number of different classes

    @return     A tuple of the overall accuracy over all classes and a list of
                the accuracies per class
    """
    overall_correct = 0.
    overall_total = 0.
    class_correct = [0.] * numClasses
    class_total = [0.] * numClasses

    with torch.no_grad():
        net.eval()
        for data in dataloader:
            images, labels = data
            _, predicted = torch.max(net(images), 1)
            correct = (predicted == labels).squeeze()
            overall_total += len(labels)
            overall_correct += correct.sum().item()
            for i in range(len(labels)):
                class_correct[labels[i]] += correct[i].item()
                class_total[labels[i]] += 1

    overall_accuracy = overall_correct / overall_total

    accuracyByClass = [0.] * numClasses
    for i in range(numClasses):
        accuracyByClass[i] = class_correct[i] / class_total[i]
    return (overall_accuracy, accuracyByClass)


def printClassAccuracy(classNames, accuracyByClass):
    for i in range(len(classNames)):
        print('Accuracy of %5s : %2d %%' %
              (classNames[i], 100 * accuracyByClass[i]))
