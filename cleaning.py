#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:53:38 2017

@author: SeanXinZhou
"""

import numpy as np
import pandas as pd

from field_names import gps_columns_conversion, imu_columns_conversion


def gps_data(df):
    """ clean the inputs from gps

    :param df: raw data dataframe
    :return: cleaned dataframe
    """
    df = df.rename(columns=gps_columns_conversion)
    df = df.sort_values(by=['t'])
    df = df.reset_index(drop=True)
    
    df['lat'] = df['lat'].replace('(null)', np.NaN)
    df['lat'] = df['lat'].replace(-1., np.NaN)
    
    df['long'] = df['long'].replace('(null)', np.NaN)
    df['long'] = df['long'].replace(-1., np.NaN)
    
    df['alt'] = df['alt'].replace('(null)', np.NaN)
    df['alt'] = df['alt'].replace(-1., np.NaN)
    
    df['course'] = df['course'].replace('(null)', np.NaN)
    df['course'] = df['course'].replace(-1., np.NaN)
    df['course'] = np.radians(df['course'])
    
    df['speed'] = df['speed'].replace('(null)', np.NaN)
    df['speed'] = df['speed'].replace(-1., np.NaN)
        
    df.index = pd.to_datetime(df['t'], unit='s')
    df = df[~df.index.duplicated()]  # drop duplicated index    
    return df


def imu_data(df):
    """ clean the inputs from imu

    :param df: raw data dataframe
    :return: cleaned dataframe
    """
    df = df.rename(columns=imu_columns_conversion)            
    df = df.sort_values(by=['t'])
    df = df.reset_index(drop=True)   
    
    df.index = pd.to_datetime(df['t'], unit='s')
    df = df[~df.index.duplicated()] # drop duplicated index
    return df
