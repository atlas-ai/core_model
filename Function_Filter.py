# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:37:42 2017

@author: seanx
"""

import pandas as pd
import timeit

def MA_Filter(df, rolling_size):
    start = timeit.default_timer()

##############################################################################################################################
#                                                          Read DataFrame                                                    #
##############################################################################################################################
        
    #Attitude
    att = df['attitude_sum'].rolling(window=rolling_size).mean().round(6)
    pitch = df['attitude_pitch(radians)'].rolling(window=rolling_size).mean().round(6)
    roll = df['attitude_roll(radians)'].rolling(window=rolling_size).mean().round(6)
    yaw = df['attitude_yaw(radians)'].rolling(window=rolling_size).mean().round(6)
    
    #Rotation
    rotx = df['rotation_rate_x(radians/s)'].rolling(window=rolling_size).mean().round(6)
    roty = df['rotation_rate_y(radians/s)'].rolling(window=rolling_size).mean().round(6)
    rotz = df['rotation_rate_z(radians/s)'].rolling(window=rolling_size).mean().round(6)

    #Gravity
    gx = df['gravity_x(G)'].rolling(window=rolling_size).mean().round(6)
    gy = df['gravity_y(G)'].rolling(window=rolling_size).mean().round(6)
    gz = df['gravity_z(G)'].rolling(window=rolling_size).mean().round(6)

    #Acceleration
    ax = df['user_acc_x(G)'].rolling(window=rolling_size).mean().round(6)
    ay = df['user_acc_y(G)'].rolling(window=rolling_size).mean().round(6)
    az = df['user_acc_z(G)'].rolling(window=rolling_size).mean().round(6)

    #Magnetic Field
    mx = df['magnetic_field_x(microteslas)'].rolling(window=rolling_size).mean().round(6)
    my = df['magnetic_field_y(microteslas)'].rolling(window=rolling_size).mean().round(6)
    mz = df['magnetic_field_z(microteslas)'].rolling(window=rolling_size).mean().round(6) 
    
    #GPS data
    gps = df[['latitude(degree)','longitude(degree)','altitude(meter)','speed(m/s)','course(degree)']]

##############################################################################################################################
#                                                         Output DataFrame                                                   #
##############################################################################################################################
    
    features=['attitude_sum','attitude_pitch(radians)','attitude_roll(radians)','attitude_yaw(radians)','rotation_rate_x(radians/s)',\
             'rotation_rate_y(radians/s)','rotation_rate_z(radians/s)','gravity_x(G)','gravity_y(G)','gravity_z(G)','user_acc_x(G)',\
             'user_acc_y(G)','user_acc_z(G)','magnetic_field_x(microteslas)','magnetic_field_y(microteslas)','magnetic_field_z(microteslas)']
    df_imu = pd.concat([att, pitch, roll, yaw, rotx, roty, rotz, gx, gy, gz, ax, ay, az, mx, my, mz], axis=1)   
    df_imu.columns = features
    
    df_ma = pd.merge(df_imu, gps, how='left', left_index=True, right_index=True).dropna(how='any')
    
    #df_ma.to_csv('ma.csv', index_label='timestamp(unix)')
    
    stop = timeit.default_timer()
    print ("Filter Run Time: %s seconds \n" % round((stop - start),2))
    
    return df_ma