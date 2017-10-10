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
    df_fc = ffc.car_acceleration(imu['rot_rate_x'], imu['rot_rate_y'], imu['rot_rate_z'],\
                            imu['user_a_x'], imu['user_a_y'], imu['user_a_z'],\
                            imu['g_x'], imu['g_y'], imu['g_z'],\
                            imu['m_x'], imu['m_y'], imu['m_z'],\
                            gps['course'], gps['speed'])    
    return df_fc


#Step 3: Apply filter to reduce sensors' noise
def apply_filter(df_fc, n_smooth):
    """ apply moving average method to reduce noise

    :param df_fc: dataframe after frame conversion process
    :param n_smooth: smoothing factor
    :return: series of processed accelerations, rotation rates, course and speed
    """
    df_smooth = df_fc.rolling(n_smooth).mean()
    acc_x, acc_y, rot_z, crs, spd = rdp.read_df(df_smooth)
    return acc_x, acc_y, rot_z, crs, spd


#Step 4: Detect events (turns and lane changes) and sudden starts or brakes
def event_detection_model(rot_z, crs, spd):    
    """ detect events (turns and lane changes)

    :param df_smooth: smoothed data in dataframe format
    :return: detected events stored in a summary dataframe
    """
    param = rdp.read_evt_param("detection_coefficients.csv")  
    df_event = fdet.event_detection(rot_z, crs, spd, param)    
    df_evt_sum = fdet.event_summary(df_event)       
    return df_evt_sum

def acc_detection_model(acc_x, crs, spd, z_threshold):
    """ detect excessive accelerations (e.g.sudden brakes)

    :param df_smooth: smoothed data in dataframe format
    :param z_threshold: parameter to control the level of acceleration
    :return: detected brakes stored in a summary dataframe
    """
    df_param = rdp.read_acc_param("acc_dec_coefficients.csv")
    df_acc_sum = fdet.excess_acc_detection(acc_x, crs, spd, df_param, z_threshold)    
    return df_acc_sum


#Step 5: Evaluate events and return scores
def evt_evaluation_model(acc_x, acc_y, spd, df_evt_sum):
    """ evaluate events and return scores

    :param df_smooth: smoothed data in dataframe format
    :param df_evt_sum: dataframe for event detection results
    :return: evaluation summary in dataframe format
    """    
    df_evt_eva = feva.event_eva(acc_x, acc_y, spd, df_evt_sum)        
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
def evaluation_summary(user_id, df_evt_eva, df_acc_eva):
    """ combine evaluation result table 
    
    :param user_id: id of the record
    :param df_evt_eva: dataframe for event evaluation
    :param df_acc_eva: dataframe for excess acceleration evaluation
    :return : evaluation summary table in dataframe format
    """    
    df_summary = feva.eva_sum(user_id, df_evt_eva, df_acc_eva)        
    return df_summary


###############################
####       Main Module      ###
###############################    

def main_work_flow():
    """ main module to detect, evaluate and summarise driving behaviour 
        for a single file 
    
    :return: summary tables
    """
    start = timeit.default_timer()
    
    folder = "/Users/Sean_Xin_Zhou/Documents/GitHub/data/Test Data/02 Data Checked/"
    pattern = folder + '*_Checked.xlsx'
    files = glob.glob(pattern)

    filename = files[1]  
    basename = os.path.splitext(os.path.basename(filename))[0]
    base_id = basename.replace("_Checked","")
    gps_sheet = "GPS"
    imu_sheet = "IMU"
    
    imu, gps = clean_data(filename, imu_sheet, gps_sheet)
    df_fc = convert_frame(imu, gps)
    acc_x, acc_y, rot_z, crs, spd = apply_filter(df_fc, n_smooth=100)
    df_evt = event_detection_model(rot_z, crs, spd)
    df_acc = acc_detection_model(acc_x, crs, spd, z_threshold=6)
    df_evt_eva = evt_evaluation_model(acc_x, acc_y, spd, df_evt)
    df_acc_eva = acc_evaluation_model(df_acc, z_threshold=6)
    df_sum = evaluation_summary(base_id, df_evt_eva, df_acc_eva)
    
    stop = timeit.default_timer()
    print ("Done for user: %s" % base_id)    
    print ("Run Time: %s seconds \n" % round((stop - start),2))
    
    return df_sum













