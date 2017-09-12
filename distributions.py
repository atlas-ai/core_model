import numpy as np


def sigmoid(t):
    """ sigmoid function

    :param t: input value
    :return: output of sigmoid function
    """
    return 1/(1 + np.e**(-t))


def predict_prob_sigmoid(x, theta):
    """ predict probability of sigmoid function

    :param x: input value
    :param theta: parameters
    :return: probability

    """
    p = sigmoid(np.dot(x, theta))
    return p