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
    gps = df[['t','lat','long','alt','speed','course']].copy(deep=True)
    gps = gps.sort_values(by=['t'])
    gps = gps.reset_index(drop=True)  
    gps.index = pd.to_datetime(gps['t'], unit='s')
    print('Input gps data length: %s' % gps.shape[0])
    gps = gps[~gps.index.duplicated()]
    gps = gps[~gps.isin(['NaN']).any(axis=1)]  
    gps = gps[~gps.isin([0.0]).any(axis=1)] 
    gps = gps[~gps.index.duplicated()]
    gps['course'].fillna(value=np.NaN, inplace=True)
    gps['course'] = np.radians(gps['course'])
    print('Clean gps data length: %s' % gps.shape[0])
    
    return gps

def imu_data(df):
    """ clean the inputs from imu

    :param df: raw data dataframe
    :return: cleaned dataframe
    """
    imu = df[['t','att_pitch','att_roll','att_yaw','rot_rate_x','rot_rate_y','rot_rate_z',\
              'g_x','g_y','g_z','user_a_x','user_a_y','user_a_z','m_x','m_y','m_z']].copy(deep=True)
    imu = imu.sort_values(by=['t'])
    imu = imu.reset_index(drop=True)       
    imu.index = pd.to_datetime(imu['t'], unit='s')
    print('Input imu data length: %s' % imu.shape[0])
    imu = imu[~imu.isin(['NaN']).any(axis=1)] 
    imu = imu[~imu.index.duplicated()]
    imu = imu.dropna(how='all')
    print('Clean imu data length: %s' % imu.shape[0])

    return imu


def sampling_control(imu, samp_rate):
    """ quality control of sample rate for frame conversion
    
    Due to inconsistent sampling rates delivered from frontend,
    infills are required to ensure the correct sampling rate 
    for frame conversion.
    
    :param imu: imu dataframe
    :param samp_rate: sample rate 
    :return: dataframe with infills to ensure constant sampling rate
    """
    imu_infill = pd.DataFrame(np.nan, index=np.arange(imu.shape[0]), columns=\
                              ['t','att_pitch','att_roll','att_yaw','rot_rate_x','rot_rate_y','rot_rate_z',\
                              'g_x','g_y','g_z','user_a_x','user_a_y','user_a_z','m_x','m_y','m_z'])
    
    sT = imu['t']-imu['t'].diff().bfill()
    interval =  imu['t'].diff().bfill()
    n = round(interval/(1/samp_rate),0).where(interval>1/samp_rate*3/2)
    dT = interval/n
    df_T = pd.concat([sT.rename('sT'),interval.rename('interval'),n.rename('n'),dT.rename('dT')],axis=1)
    df_T = df_T.dropna(axis=0,how='any')
    
    recNo = 0
    imuLen = df_T.shape[0]
    if imuLen>1:
        for i in range(imuLen):
            for j in range(int(df_T['n'][i]-1)):
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
    print('sampling rate control imu data length: %s' % df.shape[0])  
    
    return df


def apply_calibration(imu, cali_param, device_id):
    """ apply calibration process to IMU data 
    
    Rotation rates and user accelerations should be centred around zero
    when the device is static.
    Calibration process is applied to centre the values.
    
    :param imu: imu dataframe
    :param cali_coef:  
    :return: calibrated imu dataframe
    """
    df = imu.copy(deep=True)
    cali_param = cali_param[cali_param['device_id']==device_id]
    #if cali_param.empty==False:    
        #df['rot_rate_x'] = imu['rot_rate_x'] - cali_param['rot_rate_x'][0]
        #df['rot_rate_y'] = imu['rot_rate_y'] - cali_param['rot_rate_y'][0]
        #df['rot_rate_z'] = imu['rot_rate_z'] - cali_param['rot_rate_z'][0]
        #Analysis shows that acceleration is not affected by drifts
        #df['user_a_x']= imu['user_a_x'] - cali_param['user_a_x'][0]
        #df['user_a_y']= imu['user_a_y'] - cali_param['user_a_y'][0]
        #df['user_a_z']= imu['user_a_z'] - cali_param['user_a_z'][0]
    
    return df


def old_imu_clean(df):
    """
    For old data in excel format
    """
    df = df.rename(columns=imu_columns_conversion)            
    df = df.sort_values(by=['t'])
    df = df.reset_index(drop=True)       
    df.index = pd.to_datetime(df['t'], unit='s')
    df = df[~df.isin(['NaN']).any(axis=1)] 
    df = df[~df.index.duplicated()]     
    return df


def old_gps_clean(df):
    """
    For old data in excel format
    """
    df = df.rename(columns=gps_columns_conversion)
    df = df.sort_values(by=['t'])
    df = df.reset_index(drop=True)   
    df = df.fillna(value=np.NaN)    
    df['lat'] = df['lat'].replace('null', np.NaN)
    df['lat'] = df['lat'].replace('', np.NaN)
    df['lat'] = df['lat'].replace(-1., np.NaN)    
    df['long'] = df['long'].replace('null', np.NaN)
    df['long'] = df['long'].replace('', np.NaN)
    df['long'] = df['long'].replace(-1., np.NaN)    
    df['alt'] = df['alt'].replace('null', np.NaN)
    df['alt'] = df['alt'].replace('', np.NaN)
    df['alt'] = df['alt'].replace(-1., np.NaN)    
    df['course'] = df['course'].replace('null', np.NaN)
    df['course'] = df['course'].replace('', np.NaN)
    df['course'] = df['course'].replace(-1., np.NaN)
    df['course'] = np.radians(df['course'])    
    df['speed'] = df['speed'].replace('null', np.NaN)
    df['speed'] = df['speed'].replace('', np.NaN)
    df['speed'] = df['speed'].replace(-1., np.NaN)        
    df.index = pd.to_datetime(df['t'], unit='s')
    df = df[~df.isin(['NaN']).any(axis=1)] 
    df = df[~df.index.duplicated()]  # drop duplicated index   
    return df