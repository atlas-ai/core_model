# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 08:22:02 2017

@author: seanx
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def Event_Plot(idx, df_sum, df):
    
    if df_sum['Type'].iloc[idx]=='RTT':
        mark = 'Right Turn'
    elif df_sum['Type'].iloc[idx]=='LTT':
        mark = 'Left Turn'
    elif df_sum['Type'].iloc[idx]=='LCR':
        mark = 'Lane Change to Right'
    elif df_sum['Type'].iloc[idx]=='LCL':
        mark = 'Lane Change to Left'
    elif df_sum['Type'].iloc[idx]=='CRS':
        mark = 'Cruise Straight'
    elif df_sum['Type'].iloc[idx]=='CRR':
        mark = 'Cruise to Right'
    elif df_sum['Type'].iloc[idx]=='CRL':
        mark = 'Cruise to Left'
    elif df_sum['Type'].iloc[idx]=='PRK':
        mark = 'Stop'

    beg_timestamp = df_sum['Start_Timestamp'].iloc[idx]
    end_timestamp = df_sum['End_Timestamp'].iloc[idx]
    duration = round((end_timestamp - beg_timestamp),2)
    
    date_time = np.asarray(df.loc[beg_timestamp:end_timestamp].index)    
    rot_z = np.asarray(df['rotation_rate_z(radians/s)'].loc[beg_timestamp:end_timestamp])
    acc_x = np.asarray(df['user_acc_x(G)'].loc[beg_timestamp:end_timestamp])
    acc_y = np.asarray(df['user_acc_y(G)'].loc[beg_timestamp:end_timestamp])
    speed = np.asarray(df['speed(m/s)'].loc[beg_timestamp:end_timestamp])*3.6
    
    segLen = len(date_time)
    time_idx = np.zeros(segLen)
    for i in range(segLen):
        time_idx[i] = i #datetime.fromtimestamp(date_time[i]).strftime('%H:%M:%S')
    
    #Plotting
    plt.figure(figsize=(12,7))    
    Characteristics_plot = plt.subplot(121)
    Lon_force = plt.subplot(322)
    Lat_force = plt.subplot(324)
    Spd_plot = plt.subplot(326)

    #Plot to identify right turns
    Characteristics_plot.plot(time_idx, rot_z,'b', label='Z-axis Rotation')
    Characteristics_plot.set_xlabel('Time Index')
    Characteristics_plot.set_ylabel('Rotation(Radians/s)')
    Characteristics_plot.legend(loc='best')
    #Set title
    Characteristics_plot.title.set_text('%s Profile with Duration of %s seconds.' % (mark, duration))
    
    #Longitudinal force
    Lon_force.plot(time_idx, acc_y,'r')
    Lon_force.yaxis.tick_right()
    Lon_force.spines['top'].set_color('none')
    Lon_force.spines['left'].set_color('none')
    Lon_force.xaxis.set_ticks_position('bottom')
    Lon_force.spines['bottom'].set_position(('data',0))
    Lon_force.set_title('Logitudinal Acceleration(G)')
    #Lateral force
    Lat_force.plot(time_idx,acc_x,'g')
    Lat_force.yaxis.tick_right()
    Lat_force.spines['top'].set_color('none')
    Lat_force.spines['left'].set_color('none')
    Lat_force.xaxis.set_ticks_position('bottom')
    Lat_force.spines['bottom'].set_position(('data',0))
    Lat_force.set_title('Lateral Acceleration(G)')
    #Speed chart
    Spd_plot.plot(time_idx,speed,'k')
    Spd_plot.yaxis.tick_right()
    Spd_plot.spines['top'].set_color('none')
    Spd_plot.spines['left'].set_color('none')
    Spd_plot.xaxis.set_ticks_position('bottom')
    Spd_plot.set_title('Speed(km/hr)')
    
    return 0
    