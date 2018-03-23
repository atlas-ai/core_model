#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:55:29 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np


def read_evt_detection_param(csv_file):
    """ read parameters for event detection 
    
    :param csv_file: input parameter file in .csv format
    :return : parameters in array format
    """
    evt_det_param = np.asarray(pd.read_csv(csv_file, index_col=0)).transpose()
    return evt_det_param


def read_acc_detection_param(csv_file):
    """ read parameters for acceleration detection
    
    :param csv_file: input parameter file in .csv format
    :return : parameters in dataframe ormat
    """
    acc_det_param = pd.read_csv(csv_file)
    return acc_det_param   


def read_evt_evaluation_param(csv_file):
    """ read parameters for event evaluation 
    
    :param param_file: csv file
    :return: data frame of coeficients
    """
    evt_eva_param = pd.read_csv(csv_file, index_col=0)
    return evt_eva_param

def read_acc_evaluation_param(csv_file):
    """ read parameters for acc evaluation 
    
    :param param_file: csv file
    :return: data frame of coeficients
    """
    acc_eva_param = pd.read_csv(csv_file, index_col=0)
    return acc_eva_param

def read_cali_matrix(csv_file, device_id):
    """ read calibration matrix file
    
    :param csv_file: input parameter file in .csv format
    :param device_id: device id
    :return : parameters in dataframe format
    """
    df = pd.read_csv(csv_file)
    cali_param = df[df['device_id']==device_id]
    return cali_param
    
def read_code_sys(xlsx_file):
    """ read coding system file
    
    :param xlsx_file: input parameter file in .xlsx format
    :return : coding information
    """
    code = pd.read_excel(xlsx_file,index_col=0)
    return code
    
 