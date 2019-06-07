# Display data sets as images with their correct and predicted labels
# Plot learning rate and accuracies over time


def labelsToClassName(labels, classes):
    return list(map(lambda i: classes[i], labels))


def actualVsPredictedClass(actualLabels, predictedLabels, classes):
    """
    Takes two lists of image labels (actual and predicted) and a list of class names.
    Returns a list actual and predicted class names separated by a new line.
    """
    actualClasses = labelsToClassName(actualLabels, classes)
    predictedClasses = labelsToClassName(predictedLabels, classes)

    def formatString(actual, predicted):
        return '{}\n{}'.format(actual, predicted)
    return list(map(formatString, actualClasses, predictedClasses))
