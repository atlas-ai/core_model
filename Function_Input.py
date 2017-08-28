# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:24:15 2017

@author: seanx
"""
import pandas as pd
import numpy as np
import timeit
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






def input_IMU(FileName, IMU, GPS):
    start = timeit.default_timer()
    
##############################################################################################################################
#                                                        Read IMU & GPS data                                                 #
##############################################################################################################################    
    

    df_IMU = pd.read_excel(FileName, sheetname=IMU)
    df_IMU['Temp_timestamp(unix)'] = df_IMU['timestamp(unix)'].copy()
    df_IMU.set_index(['Temp_timestamp(unix)'], inplace=True)
    df_IMU.sort_index(ascending=True)

    df_GPS = pd.read_excel(FileName, sheetname=GPS)
    df_GPS.set_index(['timesstamp(unix)'], inplace=True)
    df_GPS.sort_index(ascending=True)
    df_GPS = df_GPS.reindex(df_IMU.index, method='nearest')

    df=pd.merge(df_IMU, df_GPS, how='left', left_index=True, right_index=True)
    df.set_index(['timestamp(unix)'], inplace=True)

##############################################################################################################################
#                                      Manipulate attitude data to remove inflection points                                  #
##############################################################################################################################        
    
    #Data Length
    dataLen = df.shape[0]

    #Attitude and manipulation to remove inflection points
    pitch = df['attitude_pitch(radians)']
    roll = df['attitude_roll(radians)']
    yaw = df['attitude_yaw(radians)']
    
    rev_pitch = pitch.copy()
    for i in range(1,dataLen):
        if (pitch.iloc[i]-pitch.iloc[i-1])<-6:
            rev_pitch.iloc[i:] = rev_pitch.iloc[i:]+2*np.pi
        elif (pitch.iloc[i]-pitch.iloc[i-1])>6:
            rev_pitch.iloc[i:] = rev_pitch.iloc[i:]-2*np.pi
            
    rev_roll = roll.copy()
    for i in range(1,dataLen):
        if (roll.iloc[i]-roll.iloc[i-1])<-6:
            rev_roll.iloc[i:] = rev_roll.iloc[i:]+2*np.pi
        elif (roll.iloc[i]-roll.iloc[i-1])>6:
            rev_roll.iloc[i:] = rev_roll.iloc[i:]-2*np.pi
            
    rev_yaw = yaw.copy()
    for i in range(1,dataLen):
        if (yaw.iloc[i]-yaw.iloc[i-1])<-6:
            rev_yaw.iloc[i:] = rev_yaw.iloc[i:]+2*np.pi
        elif (yaw.iloc[i]-yaw.iloc[i-1])>6:
            rev_yaw.iloc[i:] = rev_yaw.iloc[i:]-2*np.pi
        
    att = rev_pitch + rev_roll + rev_yaw

##############################################################################################################################
#                                                         Output DataFrame                                                   #
##############################################################################################################################

    df_exp_att = df[['rotation_rate_x(radians/s)','rotation_rate_y(radians/s)','rotation_rate_z(radians/s)', \
                     'gravity_x(G)','gravity_y(G)','gravity_z(G)','user_acc_x(G)','user_acc_y(G)','user_acc_z(G)',\
                     'magnetic_field_x(microteslas)','magnetic_field_y(microteslas)','magnetic_field_z(microteslas)',\
                     'latitude(degree)','longitude(degree)','altitude(meter)','speed(m/s)','course(degree)']] 
    
    features=['attitude_sum','attitude_pitch(radians)','attitude_roll(radians)','attitude_yaw(radians)']   
    df_att = pd.concat([att, rev_pitch, rev_roll, rev_yaw], axis=1)
    df_att.columns = features
    
    df_output = pd.merge(df_att, df_exp_att, how='left', left_index=True, right_index=True)
    
    #df_output.to_csv('test.csv', index_label='timestamp(unix)')
    
    stop = timeit.default_timer()
    print ("Open File: %s \nRun Time: %s seconds \n" % (FileName, round((stop - start),2)))

    return df_output


