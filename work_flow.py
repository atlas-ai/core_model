#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:34:31 2017

@author: SeanXinZhou
"""

#import numpy as np
import pandas as pd

import read_data_n_param as rdp
import cleaning as cln
import frame as frm
import detection as det
import evaluation as eva
import data_query as dqu
import filtering as fil

#import glob
#import os
import timeit

"""
Process:
1) The first step is to run the function of "read_param()" to load all parameters.
2) The second step is to run the function of "execute_algorithm()", which encapsulates all functions from #1 to #5.
   The second step is for 60-second intervals.
3) At the end of the track, a message will be sent from the front end to trigger #6 function of "track_display()".
   This function will remove all duplicates created from the overlaps of 60-second intervals, and select information
   from the result table for display at the front end.
   
The followings are the global parameters and file names for parameters:

samp_rate = 50
n_smooth = 50
tn_thr = 0.8
lc_thr = 0.6
l1_thr = 2
l2_thr = 3
l3_thr = 6
l4_thr = 12
acc_thr = 1
acc_fac = 3

device_id = 'iPad-001'
cali_file = 'calibration_matrix.csv'
evt_det_file = 'evt_detection_parameters.csv'
acc_det_file = 'acc_detection_parameters.csv'
rtt_eva_file = 'rtt_evaluation_parameters.csv'
ltt_eva_file = 'ltt_evaluation_parameters.csv'
utn_eva_file = 'utn_evaluation_parameters.csv'
lcr_eva_file = 'lcr_evaluation_parameters.csv'
lcl_eva_file = 'lcl_evaluation_parameters.csv'
acc_eva_file = 'acc_evaluation_parameters.csv'
code_file = 'code_sys.xlsx'

"""

#0: Read all parameter files at the beginning
def read_param(cali_file, evt_det_file, acc_det_file, rtt_eva_file, ltt_eva_file, utn_eva_file,\
               lcr_eva_file, lcl_eva_file, acc_eva_file, code_file):
    """ read all parameter files

    :param cali_file: file name for calibration matrix
    :param evt_det_file: file name for event detection parameters
    :param acc_det_file: file name for excess acceleration detection parameters
    :param rtt_eva_file: file name for right turn evaluation parameters
    :param ltt_eva_file: file name for left turn evaluation parameters
    :param utn_eva_file: file name for u-turn evaluation parameters
    :param lcr_eva_file: file name for lane change to right evaluation parameters
    :param lcl_eva_file: file name for lane change to left evaluation parameters
    :param acc_eva_file: file name for excess acceleration evaluation parameters
    :param code_file: file name for coding system
    :param device_id: device id for calibration matrix
    :return: all parameters
    """
    cali_param = rdp.read_cali_matrix(cali_file)
    evt_param = rdp.read_evt_detection_param(evt_det_file)
    acc_param = rdp.read_acc_detection_param(acc_det_file)
    param_rtt = rdp.read_evt_evaluation_param(rtt_eva_file)
    param_ltt = rdp.read_evt_evaluation_param(ltt_eva_file)
    param_utn = rdp.read_evt_evaluation_param(utn_eva_file)
    param_lcr = rdp.read_evt_evaluation_param(lcr_eva_file)
    param_lcl = rdp.read_evt_evaluation_param(lcl_eva_file)
    param_acc = rdp.read_acc_evaluation_param(acc_eva_file)
    code_sys = rdp.read_code_sys(code_file)    
    return cali_param, evt_param, acc_param, param_rtt, param_ltt, param_utn, param_lcr, param_lcl, param_acc, code_sys


#1: Read data file and clean imu and gps data
def clean_data(raw_data):
    """ read imu and gps data and remove invalid and duplicate data

    :param filename: input file as .xlsx
    :param imu_sheet: name of imu sheet
    :param gps_sheet: name of gps sheet
    :return: imu and gps in dataframe format
    """
    imu = cln.imu_data(raw_data)
    gps = cln.gps_data(raw_data)         
    return imu, gps


#2: Conduct frame conversion process
def convert_frame(imu, gps, samp_rate, cali_param, device_id):
    """ read cleaned imu and gps data and conduct frame converion

    :param imu: dataframe for imu
    :param gps: dataframe for gps
    :return: converted data in dataframe format
    """
    start = timeit.default_timer()
    imu_samp = cln.sampling_control(imu, samp_rate)    
    imu_cal = cln.apply_calibration(imu_samp, cali_param, device_id)
    acc_imu = frm.car_acceleration(imu_cal['rot_rate_x'], imu_cal['rot_rate_y'], imu_cal['rot_rate_z'],\
                            imu_cal['user_a_x'], imu_cal['user_a_y'], imu_cal['user_a_z'],\
                            imu_cal['g_x'], imu_cal['g_y'], imu_cal['g_z'],\
                            imu_cal['m_x'], imu_cal['m_y'], imu_cal['m_z'],\
                            gps['lat'], gps['long'], gps['alt'], gps['course'], gps['speed'])  
    acc_imu = acc_imu.dropna(how='all') 
    acc_gps = frm.car_acceleration_from_gps(acc_imu['course'], acc_imu['speed'])
    df_fc = pd.concat([acc_imu, acc_gps],axis=1)   
    stop = timeit.default_timer()
    print ('Frame Conversion Run Time: %s seconds ' % round((stop - start),2))     
    return df_fc


#3: Apply filter to reduce sensors' noise
def apply_filter(df_fc, n_smooth):
    """ apply moving average method to reduce noise

    :param df_fc: dataframe after frame conversion process
    :param n_smooth: smoothing factor
    :return: series of processed accelerations, rotation rates, course and speed
    """
    start = timeit.default_timer()
    acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps = fil.acc_adjustment(df_fc, n_smooth)
    stop = timeit.default_timer()
    print ('Filtering Run Time: %s seconds ' % round((stop - start),2))         
    return acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps


#4: Detect events (turns and lane changes) and sudden starts or brakes
def evt_detection_model(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, tn_thr, lc_thr):    
    """ detect events (turns and lane changes)
    
    :param rot_rate_z: imu rotation rate around z   
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param crs: gps course in radians
    :param spd: speed in km/hr
    :param samp_rate: sampling rate of raw data 
    :param tn_thr: turn threshold
    :return: detected events stored in a summary dataframe
    """
    start = timeit.default_timer()
    df_evt = det.event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, tn_thr, lc_thr)
    stop = timeit.default_timer()
    print ('Event Detection Run Time: %s seconds ' % round((stop - start),2))          
    return df_evt


def acc_detection_model(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr):
    """ detect excessive accelerations (e.g.sudden brakes)

    :param acc_x: imu acceleration for car moving direction   
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param crs: gps course in radians
    :param spd: speed in m/s2
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :param z_threshold: parameter to control the level of acceleration
    :return: detected sudden accelerations or brakes stored in a summary dataframe
    """
    start = timeit.default_timer()
    df_acc = det.ex_acc_detection(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr)   
    stop = timeit.default_timer()
    print ('Excess Acceleration Detection Run Time: %s seconds ' % round((stop - start),2))      
    return df_acc


#5: Evaluate events and accelerations
def evaluation_model(acc_x, acc_x_gps, acc_y, rot_z, spd, crs, df_evt, df_acc, param_rtt, param_ltt, param_utn,\
                     param_lcr, param_lcl, param_acc, samp_rate, l1_thr, l2_thr, l3_thr, l4_thr, track_id):
    """
    :param acc_x: imu acceleration for car moving direction
    :param acc_y: imu lateral force
    :param spd: speed in km/h 
    :param df_evt_sum: dataframe for event detection results
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :return: evaluation summary in dataframe format
    """  
    start = timeit.default_timer()
    df_res = eva.eva_resampling(df_evt, df_acc, acc_x, acc_x_gps, acc_y, rot_z, spd, crs, samp_rate)
    df_eva = eva.evt_n_acc_evaluation(df_res, param_rtt, param_ltt, param_utn, param_lcr, param_lcl, param_acc,\
                         samp_rate, l1_thr, l2_thr, l3_thr, l4_thr, track_id)
    stop = timeit.default_timer()
    print ('Evaluation Run Time: %s seconds ' % round((stop - start),2))         
    return df_eva


#6 Execute main algorithms (encapsulate functions #1 to #5)
def execute_algorithm(raw_data, cali_param, evt_param, acc_param, param_rtt, param_ltt, param_utn, param_lcr, param_lcl,\
                      param_acc, samp_rate, n_smooth, tn_thr, lc_thr, acc_thr, l1_thr, l2_thr, l3_thr, l4_thr, \
                      track_id, device_id='iPad-001'):
    """ execute main algorithms 
    
    :param raw_data: measurements as raw data
    :param cali_param: file name for calibration matrix
    :param evt_param: file name for event detection parameters
    :param acc_param: file name for excess acceleration detection parameters
    :param param_rtt: file name for right turn evaluation parameters
    :param param_ltt: file name for left turn evaluation parameters
    :param param_utn: file name for u-turn evaluation parameters
    :param param_lcr: file name for lane change to right evaluation parameters
    :param param_lcl: file name for lane change to left evaluation parameters
    :param param_acc: file name for excess acceleration evaluation parameters
    :param samp_rate: sampling rate of raw data 
    :param n_smooth: smoothing factor
    :param tn_thr: probability threshold for turns
    :param lc_thr: probability threshold for lane changes
    :param acc_thr: threshold of z-score that acceleration breaches
    :param l1_thr: l1 threshold for severity measurement
    :param l2_thr: l2 threshold for severity measurement
    :param l3_thr: l3 threshold for severity measurement
    :param l4_thr: l4 threshold for severity measurement
    :param track_id: track uuid 
    :param device_id: device id for calibration       
    :return : detection and evaluation result table
    """  
    start = timeit.default_timer()
    imu, gps = clean_data(raw_data)
    df_fc = convert_frame(imu, gps, samp_rate, cali_param, device_id)
    acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps = apply_filter(df_fc, n_smooth)
    df_evt = evt_detection_model(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, tn_thr, lc_thr)
    df_acc = acc_detection_model(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr)
    df_eva = evaluation_model(acc_x, acc_x_gps, acc_y, rot_z, spd, crs, df_evt, df_acc, param_rtt, param_ltt,\
    param_utn, param_lcr, param_lcl, param_acc, samp_rate, l1_thr, l2_thr, l3_thr, l4_thr, track_id)
    stop = timeit.default_timer()
    print ('Execute Algorithm Total Run Time: %s seconds' % round((stop - start),2))       
    return df_eva


#6: Generate result table for display at front end
def track_summary(df_eva, code_sys, track_id, l1_thr, l2_thr, l3_thr, l4_thr, acc_fac):
    """ selection of data for display (run at the end of 60s intervals)
    
    :param df_eva: evaluation result table
    :param code_sys: coding systems for events
    :param track_id: track uuid
    :param l1_thr: level 1 threshold for evaluation
    :param l2_thr: level 2 threshold for evaluation
    :param l3_thr: level 3 threshold for evaluation
    :param l4_thr: level 4 threshold for evaluation
    :param acc_fac: factor used for acceleration threshold
    :return : result table for display at front end
    """ 
    start = timeit.default_timer()     
    df_sum = dqu.remove_duplicates(df_eva, track_id)
    df_display = dqu.display_track_info(df_sum, code_sys, l1_thr, l2_thr, l3_thr, l4_thr, acc_fac)
    df_track = dqu.track_info_summary(df_sum, df_display)
    stop = timeit.default_timer()
    print ('Summary Run Time: %s seconds ' % round((stop - start),2))         
    return df_sum, df_display, df_track








