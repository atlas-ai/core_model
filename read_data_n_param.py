#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:55:29 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np


def get_id(filename):
    """ read input file and retrieve id 
    
    :param filename: input file in .xlsx
    :return: id
    """
    base_id = filename.replace("_Checked.xlsx","")
    return base_id


def read_df(df):
    """ read data file for event detection 
    
    :param df: input dataframe
    :return rot_z: rotation rate of z-axis
    :return crs: course in radian
    :return spd: speed of a vehicle in km/hr
    """
    #df = pd.read_excel(file_name)
    #df.set_index(['t'], inplace=True)
    df = df.dropna(how='all')
    acc_x = df['acc_x']
    acc_y = df['acc_y']
    rot_z = df['r_rate_z']
    lat = df['lat']
    long = df['long']
    alt = df['alt']
    crs = df['course']
    spd = df['speed']*3.6
    acc_x_gps = df['acc_x_gps']
    acc_y_gps = df['acc_y_gps']
    return acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps


def read_evt_param(csv_file):
    """ read parameters for event detection 
    
    :param csv_file: input parameter file in .csv format
    :return : parameters in array format
    """
    evt_param = np.asarray(pd.read_csv(csv_file, index_col=0)).transpose()
    return evt_param


def read_acc_param(csv_file):
    """ read parameters for excess acceleration
    
    :param csv_file: input parameter file in .csv format
    :return : parameters in dataframe ormat
    """
    acc_param = pd.read_csv(csv_file)
    return acc_param   


def read_eva_param(csv_file):
    """ read evaluation parameter file 
    
    :param param_file: csv file
    :return: data frame of coeficients
    """
    eva_param = pd.read_csv(csv_file, index_col=0)
    return eva_param


 