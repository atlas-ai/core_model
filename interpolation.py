import numpy as np


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


def diff_t(s):
    """ time differentiation of a time indexed series

    :param s: time indexed series
    :return: time differentiate of s on a second
    """
    res = s.diff().bfill() / s.index.to_series().diff().bfill().dt.total_seconds()
    return res


def signal_err(s1, s2):
    """ computes the norm of the difference of 2 series

    This is a simple way to caracterize the difference between the 2 series

    :param s1: pandas series
    :param s2: pandas series
    :return: norm of the difference of the 2 series
    """
    ds = s1 - s2
    res = np.sqrt(np.square(ds))
    return res
