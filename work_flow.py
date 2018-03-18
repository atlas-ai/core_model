#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:34:31 2017

@author: SeanXinZhou
"""

import numpy as np
import pandas as pd

import read_data_n_param as rdp
import cleaning as cln
import frame as frm
import detection as det
import evaluation as eva
import data_query as dqu
import filtering as fil

import glob
import os
import timeit


#Step 1: Read data file and clean imu and gps data
def clean_data(filename, imu_sheet, gps_sheet):
    """ read imu and gps data and remove invalid and duplicate data

    :param filename: input file as .xlsx
    :param imu_sheet: name of imu sheet
    :param gps_sheet: name of gps sheet
    :return: imu and gps in dataframe format
    """
    res = pd.read_excel(filename,sheetname=[imu_sheet,gps_sheet])
    imu = cln.imu_data(res[imu_sheet])
    gps = cln.gps_data(res[gps_sheet])     
    return imu, gps


#Step 2: Conduct frame conversion process
def convert_frame(imu, gps, samp_rate, cali_file, device_id):
    """ read cleaned imu and gps data and conduct frame converion

    :param imu: dataframe for imu
    :param gps: dataframe for gps
    :return: converted data in dataframe format
    """
    start = timeit.default_timer()
    cali_param = rdp.read_cali_matrix(cali_file, device_id)
    imu_samp = cln.sampling_control(imu, samp_rate)    
    imu_cal = cln.apply_calibration(imu_samp, cali_param)
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


#Step 3: Apply filter to reduce sensors' noise
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


#Step 4: Detect events (turns and lane changes) and sudden starts or brakes
def evt_detection_model(rot_z, lat, long, alt, crs, spd, evt_det_file, samp_rate, tn_thr, lc_thr):    
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
    evt_param = rdp.read_evt_detection_param(evt_det_file)
    df_evt = det.event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, tn_thr, lc_thr)
    stop = timeit.default_timer()
    print ('Event Detection Run Time: %s seconds ' % round((stop - start),2))      
    return df_evt

def acc_detection_model(acc_x, lat, long, alt, crs, spd, acc_det_file, samp_rate, acc_thr):
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
    acc_param = rdp.read_acc_detection_param(acc_det_file)
    df_acc = det.ex_acc_detection(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr)   
    stop = timeit.default_timer()
    print ('Excess Acceleration Detection Run Time: %s seconds ' % round((stop - start),2))      
    return df_acc


#Step 5: Evaluate events and accelerations
def evaluation_model(acc_x, acc_x_gps, acc_y, rot_z, spd, crs, df_evt, df_acc, rtt_eva_file, ltt_eva_file, utn_eva_file, \
                     lcr_eva_file, lcl_eva_file, acc_eva_file, samp_rate, l1_thr, l2_thr, l3_thr, l4_thr, track_id):
    """
    :param acc_x: imu acceleration for car moving direction
    :param acc_y: imu lateral force
    :param spd: speed in km/h
    :param df_evt_sum: dataframe for event detection results
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :return: evaluation summary in dataframe format
    """  
    start = timeit.default_timer()
    param_rtt = rdp.read_evt_evaluation_param(rtt_eva_file)
    param_ltt = rdp.read_evt_evaluation_param(ltt_eva_file)
    param_utn = rdp.read_evt_evaluation_param(utn_eva_file)
    param_lcr = rdp.read_evt_evaluation_param(lcr_eva_file)
    param_lcl = rdp.read_evt_evaluation_param(lcl_eva_file)
    param_acc = rdp.read_acc_evaluation_param(acc_eva_file)
    df_res = eva.eva_resampling(df_evt, df_acc, acc_x, acc_x_gps, acc_y, rot_z, spd, crs, samp_rate)
    df_eva = eva.evt_n_acc_evaluation(df_res, param_rtt, param_ltt, param_utn, param_lcr, param_lcl, param_acc,\
                         samp_rate, l1_thr, l2_thr, l3_thr, l4_thr, track_id)
    stop = timeit.default_timer()
    print ('Evaluation Run Time: %s seconds ' % round((stop - start),2))         
    return df_eva


#Step 6: Generate result table for display at front end
def track_display(df, track_id, l1_thr, l2_thr, l3_thr, l4_thr, acc_fac):
    """ selection of data for display (run at the end of 60s intervals)
    
    :param df: evaluation result table
    :param track_id: track uuid
    :param l1_thr: level 1 threshold for evaluation
    :param l2_thr: level 2 threshold for evaluation
    :param l3_thr: level 3 threshold for evaluation
    :param l4_thr: level 4 threshold for evaluation
    :param acc_fac: factor used for acceleration threshold
    :return : result table for display at front end
    """ 
    start = timeit.default_timer()     
    df_sum = dqu.remove_duplicates(df, track_id)
    df_display = dqu.display_track_info(df_sum, l1_thr, l2_thr, l3_thr, l4_thr, acc_fac)
    stop = timeit.default_timer()
    print ('Display Run Time: %s seconds ' % round((stop - start),2))         
    return df_display


#Execute main algorithms
def execute_algorithm(imu, gps, base_id, samp_rate, n_smooth, z_threshold, turn_threshold, lane_change_threshold):
    """ execute main algorithms 
    
    :param imu: imu data
    :param gps: gps data
    :param base_id: id of an unique trip
    :param samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :param n_smooth: smoothing factor
    :param z_threshold: threshold of z-score that acceleration breaches
    :return : result table in dataframe format
    """    
    start = timeit.default_timer()
    cali_param = rdp.read_cali_matrix('calibration_matrix.csv', device_id='iPad-001')
    df_fc = convert_frame(imu, gps, samp_rate, device_id='iPad-001')
    acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps = apply_filter(df_fc, n_smooth)    
    df_evt = evt_detection_model(rot_z, lat, long, alt, crs, spd, samp_rate, turn_threshold, lane_change_threshold)
    df_acc = acc_detection_model(acc_x, lat, long, alt, crs, spd, samp_rate, z_threshold)
    df_evt_eva = evt_evaluation_model(acc_x, acc_y, spd, df_evt, samp_rate)
    df_acc_eva = acc_evaluation_model(df_acc, z_threshold)
    df_sum = evaluation_summary(base_id, df_evt_eva, df_acc_eva, spd, acc_x_gps, samp_rate)  
    stop = timeit.default_timer()
    print ('Run Time: %s seconds ' % round((stop - start),2))
    return df_sum


##########################################################
####       Main Module - Simulate Loop of 60s Run      ###
##########################################################    

def work_flow_with_loop(imu, gps, base_id, samp_rate, n_smooth,\
                        z_threshold, turn_threshold, lane_change_threshold):    
    """ main module to detect, evaluate and summarise driving behaviour for a single file 
    
    :param file_num: file number in the folder
    :return: summary tables
    """
    start = timeit.default_timer()
    pre_time = timeit.default_timer()
    df_db = pd.DataFrame(np.nan, index=np.arange(0), columns=['id','type','prob','score','d','s_utc','e_utc',\
                       'event_acc','s_spd','e_spd','s_crs','e_crs','s_lat','e_lat','s_long','e_long','s_alt','e_alt',\
                       'sec1_s_spd','sec1_e_spd','sec1_spd_bin','sec1_acc_z','sec1_dec_z','sec1_lat_lt_z','sec1_lat_rt_z',\
                       'sec2_s_spd','sec2_e_spd','sec2_spd_bin','sec2_acc_z','sec2_dec_z','sec2_lat_lt_z','sec2_lat_rt_z',\
                       'sec3_s_spd','sec3_e_spd','sec3_spd_bin','sec3_acc_z','sec3_dec_z','sec3_lat_lt_z','sec3_lat_rt_z',\
                       'spd_1','spd_2','spd_3','spd_4','spd_5','spd_6','spd_7','spd_8','spd_9','spd_10',\
                       'spd_11','spd_12','spd_13','spd_14','spd_15','spd_16','spd_17','spd_18','spd_19','spd_20',\
                       'acc_x_gps_1','acc_x_gps_2','acc_x_gps_3','acc_x_gps_4','acc_x_gps_5',\
                       'acc_x_gps_6','acc_x_gps_7','acc_x_gps_8','acc_x_gps_9','acc_x_gps_10',\
                       'acc_x_gps_11','acc_x_gps_12','acc_x_gps_13','acc_x_gps_14','acc_x_gps_15',\
                       'acc_x_gps_16','acc_x_gps_17','acc_x_gps_18','acc_x_gps_19','acc_x_gps_20'])  
    
    #Simulate reading patterns (Do calculation every 60 seconds, with 15 seconds overlapping data.)
    imu = cln.sampling_control(imu, samp_rate)
    beg_rec = 0
    end_rec = 60*samp_rate
    tot_rep = (imu.shape[0]-60*samp_rate)//(45*samp_rate)+1   
    
    print('\n')
    print('Event Detection & Evaluation Total Number of Loops: %s' % tot_rep)
    
    for i in range(tot_rep): 
        
        beg_rec = i*45*samp_rate
        
        if i==(tot_rep-1):
            end_rec = imu.shape[0]-1
        else:
            end_rec = 60*samp_rate+i*45*samp_rate

        imu_segment = imu.iloc[beg_rec:end_rec]
        print('\n')  
        print('Event Detection & Evaluation Loop %s' % (i+1))
        df_segment = execute_algorithm(imu_segment, gps, base_id, samp_rate, n_smooth,\
                                       z_threshold, turn_threshold, lane_change_threshold)
        df_db = df_db.append(df_segment)
        df_db = df_db.reset_index(drop=True)
        print('Loop %s Run Time: %s seconds' % ((i+1),round(timeit.default_timer()-pre_time,2)))
        
        pre_time = timeit.default_timer()
        
    #Clean up process at the end of all runs
    df = clean_results(base_id, df_db)
    
    stop = timeit.default_timer()
    print ('\n')
    print ('Done for user: %s' % base_id)
    print ('Total Run Time: %s seconds \n' % round((stop - start),2))
    
    return df


##########################################################
####         Read Data Extrated from the Server        ###
##########################################################
    
def read_new_data(file_name, file_type, samp_rate):
    
    if file_type=='csv':
        
        file = "../core_model/test/test_data/" + file_name
        res = pd.read_csv(file, sep=';') 
        
        imu = res[['t','att_pitch','att_roll','att_yaw','rot_rate_x','rot_rate_y','rot_rate_z',\
                  'g_x','g_y','g_z','user_a_x','user_a_y','user_a_z','m_x','m_y','m_z']].copy()
        imu = imu.sort_values(by=['t'])
        imu = imu.reset_index(drop=True)       
        imu.index = pd.to_datetime(imu['t'], unit='s')
        imu = imu[~imu.index.duplicated()]
        
        gps = res[['t','lat','long','alt','speed','course']].copy()
        gps['course'] = np.radians(gps['course'])
        gps = gps.sort_values(by=['t'])
        gps = gps.reset_index(drop=True)  
        gps.index = pd.to_datetime(gps['t'], unit='s')
        gps = gps[~gps.index.duplicated()]
                
    elif file_type=='xlsx':
        
        file = "../data/Test Data/20171229/" + file_name
        res = pd.read_excel(file,sheetname=['GPS','IMU'])
        gps = cln.gps_data(res['GPS'])
        imu = cln.imu_data(res['IMU'])
    
        
    start_time = imu['t'][0]
    end_time = imu['t'][imu.shape[0]-1]
    jt_imu_t = (end_time-start_time)/60
    jt_imu_rec = imu.shape[0]/samp_rate/60
    print('IMU Timestamp Check:')    
    print('Estimated Journey Time from Beginning and Ending IMU Timestamps: %s minutes' %\
          (round(jt_imu_t,2))) 
    print('Estmated Journey Time from Number of IMU Data Entries and Sampling Rate: %s minutes' %\
          round(jt_imu_rec,2))
    
    gps = gps[~gps.isin(['NaN']).any(axis=1)]  
    gps = gps[~gps.isin([0.0]).any(axis=1)]
    start_gps = gps['t'][0]
    end_gps = gps['t'][gps.shape[0]-1]
    jt_gps_t = (end_gps-start_gps)/60
    jt_gps_rec = gps.shape[0]/60
    print('GPS Timestamp Check:')
    print('Estimated Journey Time from Beginning and Ending GPS Timestamps: %s minutes' %\
          (round(jt_gps_t,2))) 
    print('Estmated Journey Time from Number of GPS Data Entries: %s minutes' %\
          round(jt_gps_rec,2))
    
    return imu, gps
    
    
    
