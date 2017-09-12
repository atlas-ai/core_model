#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:56:26 2017

@author: SeanXinZhou
"""


import numpy as np
import pandas as pd


import cleaning as fin
import frame as ffc
import detection as fdet


import glob
import os
import timeit

start = timeit.default_timer()

folder = "/Users/Sean_Xin_Zhou/Documents/GitHub/data/Test Data/02 Data Checked/"
pattern = folder + '*_Checked.xlsx'
files = glob.glob(pattern)


def output_name(filename):
    dirname = os.path.dirname(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    base_id = basename.replace("_Checked","")
    filename_out = os.path.join(dirname ,base_id + "_acc.xlsx") # change here the extention of the output file name
    return base_id, filename_out


def write_acc(filename,n_smooth=100):
    base_id, filename_out = output_name(filename)
    gps_sheet = "GPS"
    imu_sheet = "IMU"
    #imu_sheet = base_id + "_Drive" # uncomment this line if the imu data is labeled with the workbook name
    res = pd.read_excel(filename,sheetname=[gps_sheet,imu_sheet])
    gps = fin.gps_data(res[gps_sheet])
    imu = fin.imu_data(res[imu_sheet])
    acc = ffc.car_acceleration(imu['rot_rate_x'], imu['rot_rate_y'], imu['rot_rate_z'],\
                            imu['user_a_x'], imu['user_a_y'], imu['user_a_z'],\
                            imu['g_x'], imu['g_y'], imu['g_z'],\
                            imu['m_x'], imu['m_y'], imu['m_z'],\
                            gps['course'], gps['speed'])
    acc_smooth = acc.rolling(n_smooth).mean()
    acc_smooth.to_excel(filename_out)
    return acc_smooth
    
filename = files[1]
df_smooth = write_acc(filename)


def detection_model(df):
    
    param = fdet.read_param("Detection_Coefficients.csv")
    
    rot_z, crs, spd = fdet.read_df(df)
    
    df_event = fdet.Event_Detection(rot_z, crs, spd, param)
    
    df_evt_sum = fdet.Event_Summary(df_event)
       
    return df_evt_sum

df_evt = detection_model(df_smooth)

stop = timeit.default_timer()
print ("Run Time: %s seconds \n" % round((stop - start),2))

