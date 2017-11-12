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


def z_score(val, ave, std):
    """ calculate z score 
    
    :param val: value of a number
    :param ave: mean of a sample
    :param std: standard deviation of a sample
    :return: z score of a number w.r.t to sample distribution
    """
    return((val-ave)/std)
    
    
def spd_bins(spd):
    """ set speed bins for different speeds
    
    :param spd: speed in km/hr
    :return: bin indicator from 1-5    
    """
    spdbin = 0
    if spd<15:
        spdbin = 1
    elif spd>=15 and spd<25:
        spdbin = 2
    elif spd>=25 and spd<35:
        spdbin = 3
    elif spd>=35 and spd<45:
        spdbin = 4
    elif spd>45:
        spdbin = 5
    return spdbin




