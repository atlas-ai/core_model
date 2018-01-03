#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:53:38 2017

@author: SeanXinZhou
"""

import numpy as np
import pandas as pd

from field_names import gps_columns_conversion, imu_columns_conversion
import interpolation


def gps_data(df):
    """ clean the inputs from gps

    :param df: raw data dataframe
    :return: cleaned dataframe
    """
    df = df.rename(columns=gps_columns_conversion)
    df = df.sort_values(by=['t'])
    df = df.reset_index(drop=True)
    
    df = df.fillna(value=np.NaN)
    
    df['lat'] = df['lat'].replace('null', np.NaN)
    df['lat'] = df['lat'].replace('', np.NaN)
    
    df['long'] = df['long'].replace('null', np.NaN)
    df['long'] = df['long'].replace('', np.NaN)
    
    df['alt'] = df['alt'].replace('null', np.NaN)
    df['alt'] = df['alt'].replace('', np.NaN)
    
    df['course'] = df['course'].replace('null', np.NaN)
    df['course'] = df['course'].replace('', np.NaN)
    df['course'] = np.radians(df['course'])
    
    df['speed'] = df['speed'].replace('null', np.NaN)
    df['speed'] = df['speed'].replace('', np.NaN)
        
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


def sampling_control(imu, samp_rate):
    """ quality control of sample rate for frame conversion
    
    Due to inconsistent sampling rates delivered from frontend,
    infills are required to ensure the correct sampling rate 
    for frame conversion.
    
    :param imu: imu dataframe
    :param samp_rate: sample rate 
    :return: dataframe with infills to ensure constant sampling rate
    """
    imu_infill = pd.DataFrame(np.nan, index=np.arange(samp_rate*10800), columns=imu_columns_conversion)
    imu_infill = imu_infill.rename(columns=imu_columns_conversion)
    
    sT = imu['t']-imu['t'].diff()
    interval =  imu['t'].diff()
    n = round(interval/(1/samp_rate),0).where(interval>1/samp_rate*3/2)
    dT = interval/n
    df_T = pd.concat([sT.rename('sT'),interval.rename('interval'),n.rename('n'),dT.rename('dT')],axis=1)
    df_T = df_T.dropna(axis=0,how='any')
    
    recNo = 0
    imuLen = df_T.shape[0]
    for i in range(imuLen):
        for j in range(int(df_T['n'][i])-1):
            imu_infill.iloc[recNo, imu_infill.columns.get_loc('t')] = df_T['sT'][i]+(j+1)*df_T['dT'][i]
            recNo = recNo + 1
        
    imu_infill = imu_infill.dropna(how='all')
    imu_infill.index = pd.to_datetime(imu_infill['t'], unit='s')
    imu = imu.append(imu_infill)    
    imu = imu.sort_index()
    imu = imu[~imu.index.duplicated()]
    
    df = imu.copy()
    idx = df.index
    df['t'] = interpolation.interpolate_to_index(imu['t'], idx, method='time')
    df['att_pitch'] = interpolation.interpolate_to_index(imu['att_pitch'], idx, method='time')
    df['att_roll'] = interpolation.interpolate_to_index(imu['att_roll'], idx, method='time')
    df['att_yaw'] = interpolation.interpolate_to_index(imu['att_yaw'], idx, method='time')
    df['rot_rate_x'] = interpolation.interpolate_to_index(imu['rot_rate_x'], idx, method='time')
    df['rot_rate_y'] = interpolation.interpolate_to_index(imu['rot_rate_y'], idx, method='time')
    df['rot_rate_z'] = interpolation.interpolate_to_index(imu['rot_rate_z'], idx, method='time')
    df['user_a_x'] = interpolation.interpolate_to_index(imu['user_a_x'], idx, method='time')
    df['user_a_y'] = interpolation.interpolate_to_index(imu['user_a_y'], idx, method='time')
    df['user_a_z'] = interpolation.interpolate_to_index(imu['user_a_z'], idx, method='time')
    df['g_x'] = interpolation.interpolate_to_index(imu['g_x'], idx, method='time')
    df['g_y'] = interpolation.interpolate_to_index(imu['g_y'], idx, method='time')
    df['g_z'] = interpolation.interpolate_to_index(imu['g_z'], idx, method='time')
    df['m_x'] = interpolation.interpolate_to_index(imu['m_x'], idx, method='time')
    df['m_y'] = interpolation.interpolate_to_index(imu['m_y'], idx, method='time')
    df['m_z'] = interpolation.interpolate_to_index(imu['m_z'], idx, method='time') 
       
    return df








