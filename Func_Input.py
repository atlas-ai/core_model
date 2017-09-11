#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:53:38 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import quaternion as qtr

gps_columns_conversion = {
    'timesstamp(unix)': 't',
    'latitude(degree)': 'lat',
    'longitude(degree)': 'long',
    'altitude(meter)': 'alt',
    'speed(m/s)': 'speed',
    'course(degree)': 'course'
}

imu_columns_conversion = {
    'timestamp(unix)': 't',
    'attitude_pitch(radians)': 'att_pitch',
    'attitude_roll(radians)': 'att_roll',
    'attitude_yaw(radians)': 'att_yaw',
    'rotation_rate_x(radians/s)': 'rot_rate_x',
    'rotation_rate_y(radians/s)': 'rot_rate_y',
    'rotation_rate_z(radians/s)': 'rot_rate_z',
    'gravity_x(G)': 'g_x',
    'gravity_y(G)': 'g_y',
    'gravity_z(G)': 'g_z',
    'user_acc_x(G)': 'user_a_x',
    'user_acc_y(G)': 'user_a_y',
    'user_acc_z(G)':  'user_a_z',
    'magnetic_field_x(microteslas)': 'm_x',
    'magnetic_field_y(microteslas)': 'm_y',
    'magnetic_field_z(microteslas)': 'm_z',
}


def clean_input_gps(df):
    """ clean the inputs from gps

    :param df: raw data dataframe
    :return: cleaned dataframe
    """
    new_col = pd.Series(df.columns).map(gps_columns_conversion)
    df.columns = new_col
    df['course'] = df['course'].replace({-1.: np.nan})
    df['course'] = np.radians(df['course'])
    df['speed'] = df['speed'].replace({-1.: np.nan})
    df.index = pd.to_datetime(df['t'], unit='s')
    df = df[~df.index.duplicated()]  # drop duplicated index
    return df


def clean_input_imu(df):
    """ clean the inputs from imu

    :param df: raw data dataframe
    :return: cleaned dataframe
    """
    new_col = pd.Series(df.columns).map(imu_columns_conversion)
    df.columns = new_col
    df.index = pd.to_datetime(df['t'], unit='s')
    df = df[~df.index.duplicated()]  # drop duplicated index
    return df
