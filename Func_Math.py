#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:49:31 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import quaternion as qtr


def smooth_angle(theta):
    """ smooth an angle series so it never does jump of 2*pi

    :param theta: angle in radian
    :return: smooted series
    """
    up_count = (theta.diff() > 3.14).cumsum()
    down_count = (theta.diff() < -3.14).cumsum()
    level = up_count - down_count
    res = theta - level*2*np.pi
    return res


def interpolate_to_index(s, i, **kwargs):
    """ interpolates the values of s on the index i

    :param s: series to interpolate from
    :param i: index to interpolate on
    :param kwargs: optional arguments transfered to
    :return: series indexed by i with values interpolated from s
    """
    i_s = i.to_series()
    s1 = (s.align(i_s)[0])
    s2 = s1.interpolate(**kwargs)
    res = s2[i]
    return res


def cross(q1, q2):
    """ quaternion cross product

    :param q1:
    :param q2:
    :return: cross product as a quaternion
    """
    p = q1*q2
    res = 0.5 * (p - np.conjugate(p))
    return res


def dot(q1, q2):
    """ quaternion cross product

    :param q1:
    :param q2:
    :return: dot product as a real
    """
    p = q1*q2
    res = from_quaternion(p)['s']
    return res


def to_quaternion(x, y, z, s=None):
    """ convert single coordinates into quaternion

    all inputs are pd.Series output is pd.Series

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param s: scalar coordinate, 0 if omited
    :return: series of quaternion coordinate with same index as input
    """
    if s is None:
        s = pd.Series(0.,index=x.index)
    df = pd.DataFrame({'s': s, 'x': x, 'y': y, 'z': z},
                      columns=['s', 'x', 'y', 'z'])
    val = df.values
    val_q = qtr.as_quat_array(val)
    res = pd.Series(val_q, index=df.index)
    return res


def from_quaternion(s, label=None):
    """ convert quaternion to its single components

    :param s: series of quaternions
    :param label: additional label of the columns names
    :return: dataframe containing quaternion coordinate labeled s,x,y, z
    """
    val = qtr.as_float_array(s)
    col = pd.Series(['s', 'x', 'y', 'z'])
    if label is not None:
        col = label + col
    res = pd.DataFrame(val, index=s.index, columns=col)
    return res


def diff_t(s):
    """ time differentiation of a time indexed series

    :param s: time indexed series
    :return: time differentiate of s on a second
    """
    res = s.diff() / s.index.to_series().diff().dt.seconds
    return res


def sigmoid(t):     
    """ sigmoid function
    
    :param t: input value
    :return: output of sigmoid function
    """                     
    return (1/(1 + np.e**(-t)))   

def predict_prob_sigmoid(X, theta):
    """ predict probability of sigmoid function
    
    :param X: input value
    :param theta: parameters
    :return: probability
    
    """
    p = sigmoid(np.dot(X, theta))    
    return p













