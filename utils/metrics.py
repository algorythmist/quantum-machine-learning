import numpy as np


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    return loss / len(labels)


def cross_entropy_loss(classification, expected):
    """Calculate accuracy of predictions using cross entropy loss.
    :param classification: Dictionary where keys are possible classes,
                               and values are the probability the class is chosen.
    :param expected: Correct classification of the data point.
    :return: Cross entropy loss
    """
    p = classification.get(expected)  # Prob. of correct classification
    return -np.log(p + 1e-10)
