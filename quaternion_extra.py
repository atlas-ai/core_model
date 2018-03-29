#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:49:31 2017

@author: SeanXinZhou
"""

import numpy as np
import pandas as pd
import quaternion as qtr


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
    if s.empty==False:
        val = qtr.as_float_array(s)
    else:
        val = []
    col = pd.Series(['s', 'x', 'y', 'z'])
    if label is not None:
        col = label + col
    res = pd.DataFrame(val, index=s.index, columns=col)        
    return res













