#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 08:34:31 2017

@author: SeanXinZhou
"""

import numpy as np
import pandas as pd

import read_data_n_param as rdp
import cleaning as fin
import frame as ffc
import detection as fdet
import evaluation as feva
import data_query as fdqu

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
    #gps_sheet = "GPS"
    #imu_sheet = "IMU"
    res = pd.read_excel(filename,sheetname=[imu_sheet,gps_sheet])
    imu = fin.imu_data(res[imu_sheet])
    gps = fin.gps_data(res[gps_sheet])     
    return imu, gps


#Step 2: Conduct frame conversion process
def convert_frame(imu, gps):
    """ read cleaned imu and gps data and conduct frame converion

    :param imu: dataframe for imu
    :param gps: dataframe for gps
    :return: converted data in dataframe format
    """
    acc_imu = ffc.car_acceleration(imu['rot_rate_x'], imu['rot_rate_y'], imu['rot_rate_z'],\
                            imu['user_a_x'], imu['user_a_y'], imu['user_a_z'],\
                            imu['g_x'], imu['g_y'], imu['g_z'],\
                            imu['m_x'], imu['m_y'], imu['m_z'],\
                            gps['lat'], gps['long'], gps['alt'], gps['course'], gps['speed'])  
    acc_imu = acc_imu[~acc_imu.isin(['NaN']).any(axis=1)]
    
    acc_gps = ffc.car_acceleration_from_gps(acc_imu['course'], acc_imu['speed'])
    df_fc = pd.concat([acc_imu, acc_gps],axis=1)
    
    return df_fc


#Step 3: Apply filter to reduce sensors' noise
def apply_filter(df_fc, n_smooth):
    """ apply moving average method to reduce noise

    :param df_fc: dataframe after frame conversion process
    :param n_smooth: smoothing factor
    :return: series of processed accelerations, rotation rates, course and speed
    """
    df_smooth = df_fc.rolling(n_smooth).mean()
    df_smooth = df_smooth[~df_smooth.isin(['NaN']).any(axis=1)]
    acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps = rdp.read_df(df_smooth)
    return acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps


#Step 4: Detect events (turns and lane changes) and sudden starts or brakes
def event_detection_model(rot_z, lat, long, alt, crs, spd, samp_rate):    
    """ detect events (turns and lane changes)
    
    :param rot_rate_z: imu rotation rate around z   
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param crs: gps course in radians
    :param spd: speed in m/s2
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :return: detected events stored in a summary dataframe
    """
    evt_param = rdp.read_evt_param("detection_coefficients.csv")  
    df_event = fdet.event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate)    
    df_evt_sum = fdet.event_summary(df_event)       
    return df_evt_sum

def acc_detection_model(acc_x, lat, long, alt, crs, spd, samp_rate, z_threshold):
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
    acc_param = rdp.read_acc_param("acc_dec_coefficients.csv")
    df_acc_sum = fdet.excess_acc_detection(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, z_threshold)    
    return df_acc_sum


#Step 5: Evaluate events and return scores
def evt_evaluation_model(acc_x, acc_y, spd, df_evt_sum, samp_rate):
    """ evaluate events and return scores

    :param acc_x: imu acceleration for car moving direction
    :param acc_y: imu lateral force
    :param spd: speed in km/h
    :param df_evt_sum: dataframe for event detection results
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :return: evaluation summary in dataframe format
    """    
    df_evt_eva = feva.event_eva(acc_x, acc_y, spd, df_evt_sum, samp_rate)        
    return df_evt_eva

def acc_evaluation_model(df_acc_sum, z_threshold):
    """ evaluate excess acceleration and return scores 
    
    :param df_acc_sum: dataframe for detected accelerations
    :param z_threshold: threshold of z-score that acceleration breaches
    :return : evaluation summary in dataframe format
    """
    df_acc_eva = feva.acc_eva(df_acc_sum, z_threshold)
    return df_acc_eva


#Step 6: Generate output table with scores
def evaluation_summary(user_id, df_evt_eva, df_acc_eva, spd, acc_x_gps, samp_rate):
    """ combine evaluation result table 
    
    :param user_id: id of the record
    :param df_evt_eva: dataframe for event evaluation
    :param df_acc_eva: dataframe for excess acceleration evaluation
    :param spd: speed in km/h
    :param acc_x_gps: acceleration from GPS in G
    :param samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :return : evaluation summary table in dataframe format
    """    
    df_summary = feva.eva_sum(user_id, df_evt_eva, df_acc_eva)      
    df_final = fdqu.plot_data_spd_n_acc(df_summary, spd, acc_x_gps, samp_rate) 
    return df_final

#Execute main algorithms
def execute_algorithm(imu, gps, base_id, samp_rate, n_smooth, z_threshold):
    """ execute main algorithms 
    
    :param imu: imu data
    :param gps: gps data
    :param base_id: id of an unique trip
    :param samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :param n_smooth: smoothing factor
    :param z_threshold: threshold of z-score that acceleration breaches
    :return : result table in dataframe format
    """    
    df_fc = convert_frame(imu, gps)
    acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps = apply_filter(df_fc, n_smooth)
    df_evt = event_detection_model(rot_z, lat, long, alt, crs, spd, samp_rate)
    df_acc = acc_detection_model(acc_x, lat, long, alt, crs, spd, samp_rate, z_threshold)
    df_evt_eva = evt_evaluation_model(acc_x, acc_y, spd, df_evt, samp_rate)
    df_acc_eva = acc_evaluation_model(df_acc, z_threshold)
    df_sum = evaluation_summary(base_id, df_evt_eva, df_acc_eva, spd, acc_x_gps, samp_rate)  
    return df_sum

#Clean up results table to remove duplicates
def clean_results(track_uuid, df_detected_event):
    """ clean result table at the end of the run
    
    :param track_uuid: user id
    :param df_detected_event: detected events result table
    :return : cleaned result table in dataframe format
    """
    df_cleaned = fdqu.remove_duplicates(track_uuid, df_detected_event)
    return df_cleaned



    
##########################################################
####       Main Module - Simulate Loop of 60s Run      ###
##########################################################    

def work_flow_with_loop(file_num):
    
    """ main module to detect, evaluate and summarise driving behaviour for a single file 
    
    :param file_num: file number in the folder
    :return: summary tables
    """
    start = timeit.default_timer()
    pre_time = timeit.default_timer()
    
    folder = "/Users/Sean_Xin_Zhou/Documents/GitHub/data/Test Data/02 Data Checked/"
    pattern = folder + '*_Checked.xlsx'
    files = glob.glob(pattern)

    filename = files[file_num]  
    basename = os.path.splitext(os.path.basename(filename))[0]
    base_id = basename.replace("_Checked","")
    gps_sheet = "GPS"
    imu_sheet = "IMU"
    
    imu, gps = clean_data(filename, imu_sheet, gps_sheet)
    
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
    beg_rec = 1
    end_rec = 6000
    tot_rep = (imu.shape[0]-6000)//4500+1
    
    for i in range(tot_rep):    
        if i==(tot_rep-1):
            end_rec = imu.shape[0]-1
    
        imu_segment = imu.iloc[((beg_rec-1)+i*4500):(end_rec+i*4500)]
    
        print('Event Detection & Evaluation Loop %s' % (i+1))
        df_segment = execute_algorithm(imu_segment, gps, base_id, samp_rate=100, n_smooth=100, z_threshold=6)
        df_db = df_db.append(df_segment)
        df_db = df_db.reset_index(drop=True)
        print('Loop %s Time: %s seconds' % ((i+1),round(timeit.default_timer()-pre_time,2)))
        pre_time = timeit.default_timer()
        
    #Clean up process at the end of all runs
    df = clean_results(base_id, df_db)
    
    stop = timeit.default_timer()
    print ("Done for user: %s" % base_id)
    print ("Total Run Time: %s seconds \n" % round((stop - start),2))
    
    return df


##########################################################
####         Read Data Extrated from the Server        ###
##########################################################

def read_new_data(filename, track_id):
    
    df = pd.read_csv(filename)
    df = df[df['track_uuid']==track_id]

    imu = df[['t','att_pitch','att_roll','att_yaw','rot_rate_x','rot_rate_y','rot_rate_z','g_x','g_y','g_z',\
                'user_a_x','user_a_y','user_a_z','m_x','m_y','m_z']]
     
    imu.index = pd.to_datetime(imu['t'], unit='s')
    imu = imu[~imu.index.duplicated()]
    imu.drop('t',axis=1)
    
    gps = df[['t','lat','long','alt','speed','course']]
    gps['course'] = gps['course'].replace({-1.: np.nan})
    gps['course'] = np.radians(gps['course'])
    gps['speed'] = gps['speed'].replace({-1.: np.nan})
    gps.index = pd.to_datetime(gps['t'], unit='s')
    gps = gps[~gps.index.duplicated()]  # drop duplicated index
    gps.drop('t', axis=1)

    return imu, gps
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
