#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:16:06 2017

@author: SeanXinZhou
"""
import pandas as pd
import numpy as np


def remove_duplicates(df_sum, track_id):
    """ clean result table at the end of the run for 60-second intervals
    
    :param user_id: user id
    :param df_sum: all detected results
    :return : cleaned result table in dataframe format
    """    
    df_sum = df_sum[df_sum['uuid'] == track_id]
    df = df_sum.copy(deep=True)
    df = df.replace('(null)', np.NaN)
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
    
    df = df[['uuid','type','prob','score','d','s_utc','e_utc',\
             'event_acc','s_spd','e_spd','s_crs','e_crs','s_lat','e_lat','s_long','e_long','s_alt','e_alt',\
             'sec1_s_spd','sec1_e_spd','sec1_spd_bin','sec1_acc_z','sec1_dec_z','sec1_lat_lt_z','sec1_lat_rt_z',\
             'sec2_s_spd','sec2_e_spd','sec2_spd_bin','sec2_acc_z','sec2_dec_z','sec2_lat_lt_z','sec2_lat_rt_z',\
             'sec3_s_spd','sec3_e_spd','sec3_spd_bin','sec3_acc_z','sec3_dec_z','sec3_lat_lt_z','sec3_lat_rt_z',\
             'acc_1','acc_2','acc_3','acc_4','acc_5','acc_6','acc_7','acc_8','acc_9','acc_10',\
             'acc_11','acc_12','acc_13','acc_14','acc_15','acc_16','acc_17','acc_18','acc_19','acc_20',\
             'lfc_1','lfc_2','lfc_3','lfc_4','lfc_5','lfc_6','lfc_7','lfc_8','lfc_9','lfc_10',\
             'lfc_11','lfc_12','lfc_13','lfc_14','lfc_15','lfc_16','lfc_17','lfc_18','lfc_19','lfc_20',\
             'spd_1','spd_2','spd_3','spd_4','spd_5','spd_6','spd_7','spd_8','spd_9','spd_10',\
             'spd_11','spd_12','spd_13','spd_14','spd_15','spd_16','spd_17','spd_18','spd_19','spd_20',\
             'crs_1','crs_2','crs_3','crs_4','crs_5','crs_6','crs_7','crs_8','crs_9','crs_10',\
             'crs_11','crs_12','crs_13','crs_14','crs_15','crs_16','crs_17','crs_18','crs_19','crs_20',\
             'rot_1','rot_2','rot_3','rot_4','rot_5','rot_6','rot_7','rot_8','rot_9','rot_10',\
             'rot_11','rot_12','rot_13','rot_14','rot_15','rot_16','rot_17','rot_18','rot_19','rot_20']]
    
    return df


def severity_bins(z_score, l1_thr, l2_thr, l3_thr, l4_thr, sign, factor=1):
    """ return the severity bins
    
    :param z_score: z score of a value
    :param l1_thr: level 1 threshold for evaluation
    :param l2_thr: level 2 threshold for evaluation
    :param l3_thr: level 3 threshold for evaluation
    :param l4_thr: level 4 threshold for evaluation
    :param sign: sign of z score
    :param factor: factors to adjust threholds
    :return: severity bin value from 1 to 5
    """
    sev_bin = 0.
    if z_score!=np.NaN:
        if sign=='pos':
            if z_score<(l1_thr/factor):
                sev_bin = 1.
            elif (z_score>=(l1_thr/factor)) and (z_score<(l2_thr/factor)):
                sev_bin = 2.
            elif (z_score>=(l2_thr/factor)) and (z_score<(l3_thr/factor)):
                sev_bin = 3.
            elif (z_score>=(l3_thr/factor)) and (z_score<(l4_thr/factor)):
                sev_bin = 4.
            elif z_score>=(l4_thr/factor):
                sev_bin = 5.
        elif sign=='neg':
            if z_score>-1.*(l1_thr/factor):
                sev_bin = 1.
            elif (z_score<=-1.*(l1_thr/factor)) and (z_score>-1.*(l2_thr/factor)):
                sev_bin = 2.
            elif (z_score<=-1.*(l2_thr/factor)) and (z_score>-1.*(l3_thr/factor)):
                sev_bin = 3.
            elif (z_score<=-1.*(l3_thr/factor)) and (z_score>-1.*(l4_thr/factor)):
                sev_bin = 4.
            elif z_score<=-1.*(l4_thr/factor):
                sev_bin = 5.
    return int(sev_bin)    


def evt_sec_codes(rec, l1_thr, l2_thr, l3_thr, l4_thr):
    """ section code for a event
    
    :param rec: record of a event
    :param l1_thr: level 1 threshold for evaluation
    :param l2_thr: level 2 threshold for evaluation
    :param l3_thr: level 3 threshold for evaluation
    :param l4_thr: level 4 threshold for evaluation
    :return: section codes
    """
    if rec['type']=='rtt':
        evt_type = 1
        sec_idx = [1, 2, 3]
    elif rec['type']=='ltt':
        evt_type = 2
        sec_idx = [1, 2, 3]
    elif rec['type']=='utn':
        evt_type = 3
        sec_idx = [1, 2, 3]
    elif rec['type']=='lcr':
        evt_type = 4
        sec_idx = [1, 3]
    elif rec['type']=='lcl':
        evt_type = 5
        sec_idx = [1, 3]
    
    sec_code = np.zeros(3)   
    for i in sec_idx:
        base_code = str(evt_type) + str(i) + str(int(rec['sec'+str(i)+'_spd_bin']))
        if rec['sec'+str(i)+'_e_spd']>=rec['sec'+str(i)+'_s_spd']:            
            acc_code = base_code + str(1) + \
            str(severity_bins(rec['sec'+str(i)+'_acc_z'], l1_thr, l2_thr, l3_thr, l4_thr, 'pos'))
        elif rec['sec'+str(i)+'_e_spd']<rec['sec'+str(i)+'_s_spd']:
            acc_code = base_code + str(2) + \
            str(severity_bins(rec['sec'+str(i)+'_dec_z'], l1_thr, l2_thr, l3_thr, l4_thr, 'neg'))
        lfc_lt_code = acc_code + \
        str(severity_bins(rec['sec'+str(i)+'_lat_lt_z'], l1_thr, l2_thr, l3_thr, l4_thr, 'neg'))
        sec_code[i-1] = lfc_lt_code + \
        str(severity_bins(rec['sec'+str(i)+'_lat_rt_z'], l1_thr, l2_thr, l3_thr, l4_thr, 'pos'))
        
    if (rec['type']=='lcr') or (rec['type']=='lcl'):
        sec_code[1] = 9999999
     
    return sec_code


def display_track_info(df, l1_thr, l2_thr, l3_thr, l4_thr, acc_fac):
    """ summarise results for front end
    
    :param df: evaluation results
    :param l1_thr: level 1 threshold for evaluation
    :param l2_thr: level 2 threshold for evaluation
    :param l3_thr: level 3 threshold for evaluation
    :param l4_thr: level 4 threshold for evaluation
    :param factor: factors to adjust threholds
    :return: evaluation results for display
    """    
    df_display = df.copy(deep=True)
    df_display['sec1_code']=np.NaN
    df_display['sec2_code']=np.NaN
    df_display['sec3_code']=np.NaN
    dt_len = df_display.shape[0]
    for i in range(dt_len):
        if (df_display['type'][i]=='rtt') or (df_display['type'][i]=='ltt') or (df_display['type'][i]=='utn') or\
        (df_display['type'][i]=='lcr')  or (df_display['type'][i]=='lcl'):
            sec_code = evt_sec_codes(df_display.iloc[i], l1_thr, l2_thr, l3_thr, l4_thr)
            df_display.iloc[i, df_display.columns.get_loc('sec1_code')] = sec_code[0]
            df_display.iloc[i, df_display.columns.get_loc('sec2_code')] = sec_code[1]
            df_display.iloc[i, df_display.columns.get_loc('sec3_code')] = sec_code[2]
        elif df_display['type'][i]=='exa':
            df_display.iloc[i, df_display.columns.get_loc('sec1_code')] = \
            str(1) + str(int(df_display['sec1_spd_bin'][i])) + str(severity_bins(df_display['sec1_acc_z'][i], l1_thr, l2_thr, l3_thr, l4_thr, 'pos', acc_fac))
            df_display.iloc[i, df_display.columns.get_loc('sec2_code')] = 999
            df_display.iloc[i, df_display.columns.get_loc('sec3_code')] = 999
        elif df_display['type'][i]=='exd':
            df_display.iloc[i, df_display.columns.get_loc('sec1_code')] = \
            str(2) + str(int(df_display['sec1_spd_bin'][i])) + str(severity_bins(df_display['sec1_dec_z'][i], l1_thr, l2_thr, l3_thr, l4_thr, 'neg', acc_fac))          
            df_display.iloc[i, df_display.columns.get_loc('sec2_code')] = 999
            df_display.iloc[i, df_display.columns.get_loc('sec3_code')] = 999
    
    df_display = df_display[(df_display['type']=='rtt') | (df_display['type']=='ltt') |(df_display['type']=='utn') |\
                            (df_display['type']=='lcr') | (df_display['type']=='lcl') |\
                            (((df_display['type']=='exa') | (df_display['type']=='exd')) & (df_display['score']!=100.))]
    df_display = df_display.reset_index(drop=True)
    
    df_display = df_display[['uuid','type','prob','score','d','s_utc','e_utc',\
                             'event_acc','s_spd','e_spd','s_crs','e_crs','s_lat','e_lat','s_long','e_long','s_alt','e_alt',\
                             'sec1_s_spd','sec1_e_spd','sec1_code',\
                             'sec2_s_spd','sec2_e_spd','sec2_code',\
                             'sec3_s_spd','sec3_e_spd','sec3_code',\
                             'acc_1','acc_2','acc_3','acc_4','acc_5','acc_6','acc_7','acc_8','acc_9','acc_10',\
                             'acc_11','acc_12','acc_13','acc_14','acc_15','acc_16','acc_17','acc_18','acc_19','acc_20',\
                             'lfc_1','lfc_2','lfc_3','lfc_4','lfc_5','lfc_6','lfc_7','lfc_8','lfc_9','lfc_10',\
                             'lfc_11','lfc_12','lfc_13','lfc_14','lfc_15','lfc_16','lfc_17','lfc_18','lfc_19','lfc_20',\
                             'spd_1','spd_2','spd_3','spd_4','spd_5','spd_6','spd_7','spd_8','spd_9','spd_10',\
                             'spd_11','spd_12','spd_13','spd_14','spd_15','spd_16','spd_17','spd_18','spd_19','spd_20']]       
    return df_display



