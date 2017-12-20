#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:54:58 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import quaternion_extra as qextra
import quaternion as qtr

import interpolation


def car_acceleration(rot_rate_x, rot_rate_y, rot_rate_z,
                     user_a_x, user_a_y, user_a_z,
                     g_x, g_y, g_z, m_x, m_y, m_z, 
                     lat, long, alt, course, speed):
    """ provides the rotation rate and acceleration in the car frame

    the function expect pandas series with a time index as its arguments
    All parameter from the imu supposed are to be with the same index.
    The result is indexed by the index of the imu data
    
    see the frame reference document for the meaning of the axis

    :param rot_rate_x: imu rotation rate around x
    :param rot_rate_y: imu rotation rate around y
    :param rot_rate_z: imu rotation rate around z
    :param user_a_x: imu user acceleration toward x
    :param user_a_y: imu user acceleration toward y
    :param user_a_z: imu user acceleration toward z
    :param g_x: imu gravity component toward x
    :param g_y: imu gravity component toward y
    :param g_z: imu gravity component toward z
    :param m_x: imu magnetic field component toward x
    :param m_y: imu magnetic field component toward y
    :param m_z: imu magnetic field component toward z    
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param course: gps course in radians
    :param speed: gps speed in m/s2
    :return: 
    """
    #Interpolation of all data in case of no entry
    i = rot_rate_x.index   
    course = interpolation.smooth_angle(course)
    course_imu = interpolation.interpolate_to_index(course, i, method='time') 
    lat_imu = interpolation.interpolate_to_index(lat, i, method='time')
    long_imu = interpolation.interpolate_to_index(long, i, method='time')    
    alt_imu = interpolation.interpolate_to_index(alt, i, method='time')
    speed_imu = interpolation.interpolate_to_index(speed, i, method='time')
    
    car_to_geo = course_to_frame(course_imu)

    g = qextra.to_quaternion(g_x, g_y, g_z)
    m = qextra.to_quaternion(m_x, m_y, m_z)
    phone_to_geo = mag_grav_to_frame(g, m)

    phone_to_car = np.invert(car_to_geo) * phone_to_geo

    user_a_phone = qextra.to_quaternion(user_a_x, user_a_y, user_a_z)
    user_a_car = phone_to_car * user_a_phone * np.invert(phone_to_car)
    user_a_car_v = qextra.from_quaternion(user_a_car, label='acc_')
    # we drop the scalar part it contains no information (always 0)
    user_a_car_v.drop(['acc_s'], inplace=True, axis=1)

    r_rate_phone = qextra.to_quaternion(rot_rate_x, rot_rate_y, rot_rate_z)
    r_rate_car = phone_to_car * r_rate_phone * np.invert(phone_to_car)
    r_rate_car_v = qextra.from_quaternion(r_rate_car, label='r_rate_')
    r_rate_car_v.drop(['r_rate_s'],inplace=True, axis=1)
    
    #The minus sign in the acceleration comes from the fact that 
    #the imu perceive a force that is in the opposite direction of the car acceleration
    res = pd.concat([user_a_car_v, r_rate_car_v],axis = 1)
    res['acc_x']=-1.0*res['acc_x']
    res['acc_y']=-1.0*res['acc_y']
    res['lat'] = lat_imu 
    res['long'] = long_imu   
    res['alt'] = alt_imu
    res['course'] = course_imu
    res['speed'] = speed_imu
    return res


def car_acceleration_from_gps(course,speed, g=9.8):
    """ Provide the acceleration and rotation rate of the car from the gps data only

    To use the GPS data, we consider that the car is a rigid body that can only rotate along the `z` axis

    we have
    - r_rate_x_gps $ \sim 0 $
    - r_rate_y_gps $ \sim 0 $
    - r_rate_z_gps = $ d_{t}(course) $

    where $d_{t}$ is the time derivative

    The acceleration can be derived from curvilinear coordinates
    that is

    $\vec{v}= \dot{s}\vec{e_t}$

    $\vec{a}= \ddot{s}\vec{e_t}+\dot{s}\dot{\theta}\vec{e_n}$

    where $s$ is the curvilinear coordinate , $\vec{e_t}$ the tangantial vector
    to the trajectory, $\vec{e_n}$ the normal vector to the trajectory,
    $\theta$ the angle of the trajectory with a given direction and $\dot{()}$ is the time derivative

    Back to our notation where

    $course = \theta$

    $\vec{e_t} = \vec{e_x}$

    $\vec{e_n} = \vec{e_y}$

    $\dot{s} = speed$

    This gives

    - $acc\_x\_gps =  d_{t}(speed) / g $
    - $acc\_y\_gps =  speed \times  d_{t}(theta) / g$
    - $acc\_z\_gps \sim 0  $

    Where
    - g is the earth gravity acceleration 9.8 m/s^2 so the acceleration units
      are coherent with imu data
    - signs are corrected to make sure both gps and imu accelerations are in line with car accelerations  
    
    With those formula it is then easy to compare gps and imu data in the car frame

    :param course: gps course in radian
    :param speed:  gps speed in m/s
    :param g: acceleration gravity in m/s^2
    :return: acceleration and rotation rates in the car frame
    """

    r_rate_z_gps = interpolation.diff_t(course)
    acc_x_gps = interpolation.diff_t(speed) / g
    acc_y_gps = r_rate_z_gps * speed /g
    res = pd.DataFrame(
        {'acc_x_gps':acc_x_gps, 'acc_y_gps':acc_y_gps, 'acc_z_gps': 0,
         'r_rate_x_gps': 0, 'r_rate_y_gps': 0, 'r_rate_z_gps': r_rate_z_gps},
        columns=['acc_x_gps', 'acc_y_gps', 'acc_z_gps', 'r_rate_x_gps',
                 'r_rate_y_gps', 'r_rate_z_gps']
    )
    return res


def course_to_frame(course):
    """ provides the geoframe of a car given the gps course angle

    We assumes that the car is standing on its wheels.


    :param course: clockwise angle from the north of the gps course in radian
    :return: frame as a quaternion
    """
    # we directly use the quaternion definition rotation
    # with a rotation in the z direction (cf geoframe definition)
    # and an angle given by the gps course
    # r = exp(course/2*k)
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    return np.exp(course/2 * qtr.quaternion(0, 0, 0, 1))


def ypr_to_frame(yawn, pitch, roll):
    """    Generate unit quaternion frame from yawn, pitch and roll

    yawn -> rotation around z axis
    pitch -> rotation around x axis
    roll -> rotation around y axis

    the rotation are applied in this order:

     yawn -> pitch -> roll

    the formula used is `exp(yawn*z/2) * exp(pitch*x/2) * exp(roll*y/2)`


    :param yawn: alpha in radian
    :param pitch: beta in radian
    :param roll: gamma in radian
    :return: frame as a quaternion
    """

    r_yawn = np.exp(yawn/2 * qtr.quaternion(0, 0, 0, 1))
    r_pitch = np.exp(pitch/2 * qtr.quaternion(0, 1, 0, 0))
    r_roll = np.exp(roll/2 * qtr.quaternion(0, 0, 1, 0))
    res = r_yawn * r_pitch * r_roll
    return res


def rotate_vec(v_origin, v_dest):
    """ Generate quaternion rotating v_origin to v_dest

    This is not uniquely defined,
    we return the rotation axis providing the shorter angle

    This function is based on quaternion algebra only

    it uses the fact that when v_1 and v_2 are unit pure vector

    ```v_1 * v_2 = -cos(theta) + sin(theta) v_3```

    where v_3 is the unit rotation vector from v_1 to v_2
    and that the rotation vector from v_1 to v_2 is

     ```exp(theta/2 v_3)=  cos(theta/2) + sin(theta/2) v_3```



    :param v_origin: origin vector as quaternion
    :param v_dest: destination vector as quaternion
    :return: rotation quaternion
    """
    # other potential approach
    # http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors

    v_origin = v_origin / np.absolute(v_origin)
    v_dest = v_dest / np.absolute(v_dest)

    res = np.exp(0.5*np.log(- np.conjugate(v_origin * v_dest)))

    return res


def rotate_2vec(v_origin1, v_dest1, v_origin2, v_dest2):
    """ Generate quaternion rotating v_origin1 to v_dest1 and v_origin2 to v_dest2

    works if v_origin1 and v_origin2 are orthogonal and so are v_dest1 and v_dest2

    :param v_origin1: origin vector 1 q as quaternion
    :param v_dest1: destination vector 1 as quaternion
    :param v_origin2: origin vector 2 q as quaternion
    :param v_dest2: destination vector 2 as quaternion
    :return: rotation quaternion
    """

    r1 = rotate_vec(v_origin1, v_dest1)

    v_o2 = r1 * v_origin2 * np.conjugate(r1)

    r2 = rotate_vec(v_o2, v_dest2)

    res = r2 * r1

    return res


def mag_grav_to_frame(g, m):
    """ Provides the geoframe of the imu given magnetic field and gravity

    :param g: gravity as a quaternion
    :param m: magnetic field as a quaternion
    :return: frame as a quaternion
    """
    # gravity toward z in the geoframe
    y = qextra.cross(g, m)  # y direction in geoframe normal to g and m
    res = rotate_2vec(g, qtr.z, y, qtr.y)
    return res