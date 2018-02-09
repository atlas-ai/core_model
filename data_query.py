#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:16:06 2017

@author: SeanXinZhou
"""

import numpy as np


def plot_data_spd_n_acc(df, spd, acc_x_gps, samp_rate):
    """ return information to plot speed and acceleraiton charts
    
    :param df: dataframe for detected events
    :param spd: speed from GPS in m/s2
    :param acc_x_gps: acceleration from GPS in G
    :param samp_rate: sampling rate
    :return : dataframe result table
    """
    dataLen=df.shape[0]   

    for i in range (20): 
        df['spd_'+str(i+1)]=np.NaN
        df['acc_x_gps_'+str(i+1)]=np.NaN
    
    for i in range(dataLen):
        
        beg_idx = spd.index.searchsorted(df['s_utc'][i]) 
        end_idx = spd.index.searchsorted(df['e_utc'][i])
        stepSize = (end_idx-beg_idx+1)/20
        
        for j in range (20):
            df.iloc[i, df.columns.get_loc('spd_'+str(j+1))] = spd[int(np.floor(beg_idx+j*stepSize))]
            df.iloc[i, df.columns.get_loc('acc_x_gps_'+str(j+1))] = acc_x_gps[int(np.floor(beg_idx+j*stepSize))]
        
    return df


def remove_duplicates(user_id, df_sum):
    """ clean result table at the end of the run
    
    :param user_id: user id
    :param df_sum: detected events result table
    :return : cleaned result table in dataframe format
    """
    df = df_sum.replace('(null)', np.NaN)
    df = df[df['id'] == user_id]
    df = df[df['d']>0]
    df = df.sort_values(['s_utc','prob'], ascending=[True,False])
    df = df.drop_duplicates(['type','s_utc'])
    df = df.reset_index(drop=True) 
    df['duplicate']=np.NaN
    dfLen = df.shape[0]
    for i in range(1, dfLen):
        if df['type'][i]==df['type'][i-1]:
            if (np.abs((df['s_utc'][i]-df['s_utc'][i-1]).total_seconds())<=5) or (df['s_utc'][i]<=df['e_utc'][i-1]):
                df.iloc[i, df.columns.get_loc('duplicate')] = 1    
    df = df[df['duplicate']!=1.0]
    df = df.drop('duplicate', axis=1)
    df = df.reset_index(drop=True)
    
    df = df[['id','type','prob','score','d','s_utc','e_utc',\
             'event_acc','s_spd','e_spd','s_crs','e_crs','s_lat','e_lat','s_long','e_long','s_alt','e_alt',\
             'sec1_s_spd','sec1_e_spd','sec1_spd_bin','sec1_acc_z','sec1_dec_z','sec1_lat_lt_z','sec1_lat_rt_z',\
             'sec2_s_spd','sec2_e_spd','sec2_spd_bin','sec2_acc_z','sec2_dec_z','sec2_lat_lt_z','sec2_lat_rt_z',\
             'sec3_s_spd','sec3_e_spd','sec3_spd_bin','sec3_acc_z','sec3_dec_z','sec3_lat_lt_z','sec3_lat_rt_z',\
             'spd_1','spd_2','spd_3','spd_4','spd_5','spd_6','spd_7','spd_8','spd_9','spd_10',\
             'spd_11','spd_12','spd_13','spd_14','spd_15','spd_16','spd_17','spd_18','spd_19','spd_20',\
             'acc_x_gps_1','acc_x_gps_2','acc_x_gps_3','acc_x_gps_4','acc_x_gps_5',\
             'acc_x_gps_6','acc_x_gps_7','acc_x_gps_8','acc_x_gps_9','acc_x_gps_10',\
             'acc_x_gps_11','acc_x_gps_12','acc_x_gps_13','acc_x_gps_14','acc_x_gps_15',\
             'acc_x_gps_16','acc_x_gps_17','acc_x_gps_18','acc_x_gps_19','acc_x_gps_20']]  
    
    return df

