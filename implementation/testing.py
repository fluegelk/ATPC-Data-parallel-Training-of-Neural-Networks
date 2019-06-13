import torch


def computeAverageLoss(net, dataloader, lossCriterion):
    summedLoss = 0.
    with torch.no_grad():
        net.eval()
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            loss = lossCriterion(outputs, labels)
            summedLoss += loss.item()
    return summedLoss / len(dataloader)


def computeAccuracy(net, dataloader, numClasses=None):
    """
    @brief      Test a given (trained) model on a set of test data. Determine total accuracy and accuracy per class.

    @param      net         The trained model
    @param      dataloader  The data loader for the test data
    @param      numClasses  The number of different classes

    @return     A tuple of the overall accuracy over all classes and a list of the accuracies per class
    """
    overall_correct = 0.
    overall_total = 0.
    if numClasses != None:
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
            if numClasses == None:
                continue
            for i in range(len(labels)):
                class_correct[labels[i]] += correct[i].item()
                class_total[labels[i]] += 1

    overall_accuracy = overall_correct / overall_total
    if numClasses == None:
        return overall_accuracy

    accuracyByClass = [0.] * numClasses
    for i in range(numClasses):
        accuracyByClass[i] = class_correct[i] / class_total[i]
    return (overall_accuracy, accuracyByClass)


def printClassAccuracy(classNames, accuracyByClass):
    for i in range(len(classNames)):
        print('Accuracy of %5s : %2d %%' % (classNames[i], 100 * accuracyByClass[i]))
