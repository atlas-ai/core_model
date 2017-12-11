#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:45:33 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import read_data_n_param as rdp
import distributions as dst


def get_event_zscore(dtype, s_utc, e_utc, acc_x, acc_y, spd, samp_rate):
    """ evaluation model for a single event 
    
    :param dtype: type of event (right turn, left turn, lane change to right, lane change to left)
    :param s_utc: starting utc time of a event
    :param e_utc: ending utc time of a event
    :param acc_x: longitudinal force of a vehicle (all data stream)
    :param acc_y: lateral force of a vehicle (all data stream)
    :param spd: speed of a vehicle in km/hr (all data stream)
    :param samp_rate: sampling rate of data collection
    :return: speed bin, z score of acceleration, z score of deceleration, 
    z score of lateral force when making left turn, z score of lateral force when making a right turn
    for three sections (entering, in the middle, exiting)
    """
        
    if dtype=='RTT':
        evt_id = 'rtt'
        coef = rdp.read_eva_param('rtt_evaluation_parameters.csv')
        dt_num = 3
        sec_idx = [0,4,13,19]
    
    elif dtype=='LTT':
        evt_id = 'ltt'
        coef = rdp.read_eva_param('ltt_evaluation_parameters.csv')
        dt_num = 3
        sec_idx = [0,4,13,19]
        
    elif dtype=='LCR':
        evt_id = 'lcr'
        coef = rdp.read_eva_param('lcr_evaluation_parameters.csv')
        dt_num = 2
        sec_idx = [0,9,19]
        
    elif dtype=='LCL':
        evt_id = 'lcl'
        coef = rdp.read_eva_param('lcl_evaluation_parameters.csv')
        dt_num = 2
        sec_idx = [0,9,19]
        
    s_idx = acc_x.index.searchsorted(s_utc)
    e_idx = acc_x.index.searchsorted(e_utc)
    stepSize = int(e_idx-s_idx+1)//20
    
    sec_s_spd = np.zeros(dt_num)
    sec_e_spd = np.zeros(dt_num)
    sec_spd = np.zeros(dt_num)
    spd_bin = np.zeros(dt_num)
    acc_z = np.zeros(dt_num)
    dec_z = np.zeros(dt_num)
    lat_lt_z = np.zeros(dt_num)
    lat_rt_z = np.zeros(dt_num)
    
    for i in range(dt_num):
        
        #average speed
        sec_s_spd[i] = spd.loc[spd.index[s_idx+sec_idx[i]*stepSize]]
        sec_e_spd[i] = spd.loc[spd.index[s_idx+sec_idx[i+1]*stepSize-1]]
        sec_spd[i] = spd.loc[spd.index[s_idx+sec_idx[i]*stepSize]:spd.index[s_idx+sec_idx[i+1]*stepSize-1]].mean()
        
        #speed bin subject to average speed
        spd_bin[i] = dst.spd_bins(sec_spd[i])
        
        #z score for acceleration        
        acc_z[i] = dst.z_score(acc_x.loc[acc_x.index[s_idx+sec_idx[i]*stepSize]:acc_x.index[s_idx+sec_idx[i+1]*stepSize-1]].where(acc_x>0).max(),\
                      coef[evt_id+'_sec'+str(i+1)+'_acc_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_id+'_sec'+str(i+1)+'_acc_var'].iloc[(spd_bin[i]-1).astype(int)]))
        #z score for deceleration        
        dec_z[i] = dst.z_score(acc_x.loc[acc_x.index[s_idx+sec_idx[i]*stepSize]:acc_x.index[s_idx+sec_idx[i+1]*stepSize-1]].where(acc_x<0).min(),\
                      coef[evt_id+'_sec'+str(i+1)+'_dec_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_id+'_sec'+str(i+1)+'_dec_var'].iloc[(spd_bin[i]-1).astype(int)]))
        #z score for lateral force (left turn, tilting to the right)        
        lat_lt_z[i] = dst.z_score(acc_y.loc[acc_y.index[s_idx+sec_idx[i]*stepSize]:acc_y.index[s_idx+sec_idx[i+1]*stepSize-1]].where(acc_y>0).max(),\
                      coef[evt_id+'_sec'+str(i+1)+'_lat_lt_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_id+'_sec'+str(i+1)+'_lat_lt_var'].iloc[(spd_bin[i]-1).astype(int)]))
        #z score for lateral force (right turn, tilting to the left)        
        lat_rt_z[i] = dst.z_score(acc_y.loc[acc_y.index[s_idx+sec_idx[i]*stepSize]:acc_y.index[s_idx+sec_idx[i+1]*stepSize-1]].where(acc_y<0).min(),\
                      coef[evt_id+'_sec'+str(i+1)+'_lat_rt_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_id+'_sec'+str(i+1)+'_lat_rt_var'].iloc[(spd_bin[i]-1).astype(int)]))
            
    return sec_s_spd, sec_e_spd, spd_bin, acc_z, dec_z, lat_lt_z, lat_rt_z


def individual_scoring(acc_z, dec_z, lat_lt_z, lat_rt_z):
    """ calculate individual event score 
    
    :param acc_z: z score for acceleration
    :param dec_z: z score for deceleration
    :param lat_lt_z: z score for lateral force when makeing a left turn (tilting to right)
    :param lat_rt_z: z score for lateral force when makeing a right turn (tilting to left)
    :return tot_score: total score of the event
    :return z_score_matrix: individual scores for each item in matrix format
    """
    warning_threshold = 3
    alert_threshold = 6
    warning_score = 5
    alert_score = 10
    
    acc_warning_score = np.where((dst.compare_nan_array(np.greater, acc_z, warning_threshold)) &\
                                 (dst.compare_nan_array(np.less, acc_z, alert_threshold)),\
                                 (acc_z/warning_threshold-1)*warning_score , 0)
    acc_alert_score = np.where((dst.compare_nan_array(np.greater_equal, acc_z, alert_threshold)),\
                                (acc_z/alert_threshold-1)*alert_score+(alert_score-warning_score) , 0)
    
    dec_warning_score = np.where((dst.compare_nan_array(np.less, dec_z, -warning_threshold)) &\
                                 (dst.compare_nan_array(np.greater, dec_z, -alert_threshold)),\
                                 (-dec_z/warning_threshold-1)*warning_score , 0)    
    dec_alert_score = np.where((dst.compare_nan_array(np.less_equal, dec_z, -alert_threshold)),\
                               (-dec_z/alert_threshold-1)*alert_score+(alert_score-warning_score) , 0)
    
    lat_lt_warning_score = np.where((dst.compare_nan_array(np.greater, lat_lt_z, warning_threshold)) &\
                                    (dst.compare_nan_array(np.less, lat_lt_z, alert_threshold)),\
                                    (lat_lt_z/warning_threshold-1)*warning_score , 0)    
    lat_lt_alert_score = np.where((dst.compare_nan_array(np.greater_equal, lat_lt_z, alert_threshold)),\
                                  (lat_lt_z/alert_threshold-1)*alert_score+(alert_score-warning_score) , 0)
    
    lat_rt_warning_score = np.where((dst.compare_nan_array(np.less, lat_rt_z, -warning_threshold)) &\
                                    (dst.compare_nan_array(np.greater, lat_rt_z, -alert_threshold)),\
                                    (-lat_rt_z/warning_threshold-1)*warning_score , 0)
    lat_rt_alert_score = np.where((dst.compare_nan_array(np.less_equal, lat_rt_z, -alert_threshold)),\
                                  (-lat_rt_z/alert_threshold-1)*alert_score+(alert_score-warning_score) , 0)
    
    tot_score = round((acc_warning_score + acc_alert_score + dec_warning_score + dec_alert_score + \
    lat_lt_warning_score + lat_lt_alert_score + lat_rt_warning_score + lat_rt_alert_score).sum(),0)
    
    z_score_matrix = np.column_stack((acc_warning_score, acc_alert_score, dec_warning_score, dec_alert_score, \
    lat_lt_warning_score, lat_lt_alert_score, lat_rt_warning_score, lat_rt_alert_score)).transpose()
    
    return tot_score, z_score_matrix
    
    
def event_eva(acc_x, acc_y, spd, df_evt, samp_rate):
    """ output event evaluation result table 
    
    :param acc_x: longitudinal force of a vehicle (all data stream)
    :param acc_y: lateral force of a vehicle (all data stream)
    :param spd: speed of a vehicle in km/hr (all data stream)
    :param df_evt: event table from detection model
    :param samp_rate: sampling rate of data collection
    :return : evaluation summary table in dataframe format
    """
    
    df_evaluation = pd.DataFrame(np.nan, index=np.arange(10000), columns=['type','prob','score','d','s_utc','e_utc',\
                       'event_acc','s_spd','e_spd','s_crs','e_crs','s_lat','e_lat','s_long','e_long','s_alt','e_alt',\
                       'sec1_s_spd','sec1_e_spd','sec1_spd_bin','sec1_acc_z','sec1_dec_z','sec1_lat_lt_z','sec1_lat_rt_z',\
                       'sec2_s_spd','sec2_e_spd','sec2_spd_bin','sec2_acc_z','sec2_dec_z','sec2_lat_lt_z','sec2_lat_rt_z',\
                       'sec3_s_spd','sec3_e_spd','sec3_spd_bin','sec3_acc_z','sec3_dec_z','sec3_lat_lt_z','sec3_lat_rt_z'])  

    evt_num = df_evt.shape[0]

    for i in range(evt_num):    
        sec_s_spd, sec_e_spd, spd_bin, acc_z, dec_z, lat_lt_z, lat_rt_z = \
        get_event_zscore(df_evt['type'][i], df_evt['s_utc'][i], df_evt['e_utc'][i], acc_x, acc_y, spd, samp_rate)
        tot_score, z_score_matrix = individual_scoring(acc_z, dec_z, lat_lt_z, lat_rt_z)
    
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('type')] = df_evt['type'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('prob')] = df_evt['prob'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('score')] = 100.0-tot_score
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('d')] = df_evt['d'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('s_utc')] = df_evt['s_utc'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('e_utc')] = df_evt['e_utc'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('event_acc')] = df_evt['event_acc'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('s_spd')] = df_evt['s_spd'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('e_spd')] = df_evt['e_spd'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('s_crs')] = df_evt['s_crs'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('e_crs')] = df_evt['e_crs'][i]  
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('s_lat')] = df_evt['s_lat'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('e_lat')] = df_evt['e_lat'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('s_long')] = df_evt['s_long'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('e_long')] = df_evt['e_long'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('s_alt')] = df_evt['s_alt'][i]
        df_evaluation.iloc[i, df_evaluation.columns.get_loc('e_alt')] = df_evt['e_alt'][i]
            
        if (df_evt['type'][i]=='RTT') or (df_evt['type'][i])=='LTT':
            sec_num=3
        elif (df_evt['type'][i]=='LCR') or (df_evt['type'][i])=='LCL':
            sec_num=2
    
        for j in range(sec_num):
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_s_spd')] = sec_s_spd[j]
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_e_spd')] = sec_e_spd[j]
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_spd_bin')] = spd_bin[j]
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_acc_z')] = acc_z[j]
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_dec_z')] = dec_z[j]
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_lat_lt_z')] = lat_lt_z[j]
            df_evaluation.iloc[i, df_evaluation.columns.get_loc('sec'+str(j+1)+'_lat_rt_z')] = lat_rt_z[j]
            
    df_evaluation = df_evaluation.dropna(how='all') 
    
    return df_evaluation
    

def acc_eva(df_acc, z_threshold):
    """ output excess acceleration evaluation result table 
    
    :param df_acc: dataframe for detected accelerations
    :param z_threshold: threshold of z-score that acceleration breaches
    :return : evaluation summary table in dataframe format
    """
    df_acc_eva = df_acc.copy(deep=True)
    df_acc_eva['score']=0.0
    df_acc_eva = df_acc_eva[['type','prob','score','d','s_utc','e_utc',\
                             'event_acc','s_spd','e_spd','s_crs','e_crs',\
                             's_lat','e_lat','s_long','e_long','s_alt','e_alt']]
    param = rdp.read_acc_param("acc_dec_coefficients.csv")
    acc_num = df_acc_eva.shape[0]
    alert_score = 3
    
    for i in range(acc_num):
        if df_acc['event_acc'][i]>0:
            acc_score = dst.z_score(df_acc['event_acc'][i],param['acc_ave'][0],np.sqrt(param['acc_var'][0]))
            df_acc_eva.iloc[i, df_acc_eva.columns.get_loc('score')] = 100.0-round((acc_score-z_threshold)*alert_score,0)
        elif df_acc['event_acc'][i]<0:
            acc_score = dst.z_score(df_acc['event_acc'][i],param['dec_ave'][0],np.sqrt(param['dec_var'][0]))            
            df_acc_eva.iloc[i, df_acc_eva.columns.get_loc('score')] = 100.0-round(-1*(acc_score+z_threshold)*alert_score,0)

    return df_acc_eva


def eva_sum(user_id, df_evt_eva, df_acc_eva):
    """ combine evaluation result table 
    
    :param user_id: id of the record
    :param df_evt_eva: dataframe for event evaluation
    :param df_acc_eva: dataframe for excess acceleration evaluation
    :return : evaluation summary table in dataframe format
    """
    evt_len = df_evt_eva.shape[0]
    acc_len = df_acc_eva.shape[0]
    tot_len = evt_len + acc_len 
    
    df_summary = pd.DataFrame(np.nan, index=np.arange(tot_len), columns=['id','type','prob','score','d','s_utc','e_utc',\
                       'event_acc','s_spd','e_spd','s_crs','e_crs','s_lat','e_lat','s_long','e_long','s_alt','e_alt',\
                       'sec1_s_spd','sec1_e_spd','sec1_spd_bin','sec1_acc_z','sec1_dec_z','sec1_lat_lt_z','sec1_lat_rt_z',\
                       'sec2_s_spd','sec2_e_spd','sec2_spd_bin','sec2_acc_z','sec2_dec_z','sec2_lat_lt_z','sec2_lat_rt_z',\
                       'sec3_s_spd','sec3_e_spd','sec3_spd_bin','sec3_acc_z','sec3_dec_z','sec3_lat_lt_z','sec3_lat_rt_z'])  
    
    rec_num = 0
    for i in range(evt_len):
        df_summary.iloc[rec_num, df_summary.columns.get_loc('id')] = user_id
        df_summary.iloc[rec_num, df_summary.columns.get_loc('type')] = df_evt_eva['type'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('prob')] = df_evt_eva['prob'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('score')] = df_evt_eva['score'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('d')] = (df_evt_eva['e_utc'][i]-df_evt_eva['s_utc'][i]).total_seconds()
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_utc')] = df_evt_eva['s_utc'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_utc')] = df_evt_eva['e_utc'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('event_acc')] = df_evt_eva['event_acc'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_spd')] = df_evt_eva['s_spd'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_spd')] = df_evt_eva['e_spd'][i]        
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_crs')] = df_evt_eva['s_crs'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_crs')] = df_evt_eva['e_crs'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_lat')] = df_evt_eva['s_lat'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_lat')] = df_evt_eva['e_lat'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_long')] = df_evt_eva['s_long'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_long')] = df_evt_eva['e_long'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_alt')] = df_evt_eva['s_alt'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_alt')] = df_evt_eva['e_alt'][i]
        
        for j in range(3):
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_s_spd')] = df_evt_eva['sec'+str(j+1)+'_s_spd'][i]
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_e_spd')] = df_evt_eva['sec'+str(j+1)+'_e_spd'][i]
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_spd_bin')] = df_evt_eva['sec'+str(j+1)+'_spd_bin'][i]
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_acc_z')] = df_evt_eva['sec'+str(j+1)+'_acc_z'][i]
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_dec_z')] = df_evt_eva['sec'+str(j+1)+'_dec_z'][i]
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_lat_lt_z')] = df_evt_eva['sec'+str(j+1)+'_lat_lt_z'][i]
            df_summary.iloc[i, df_summary.columns.get_loc('sec'+str(j+1)+'_lat_rt_z')] = df_evt_eva['sec'+str(j+1)+'_lat_rt_z'][i]        
        rec_num += 1
        
    for i in range(acc_len):
        df_summary.iloc[rec_num, df_summary.columns.get_loc('id')] = user_id
        df_summary.iloc[rec_num, df_summary.columns.get_loc('type')] = df_acc_eva['type'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('prob')] = df_acc_eva['prob'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('score')] = df_acc_eva['score'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('d')] = (df_acc_eva['e_utc'][i]-df_acc_eva['s_utc'][i]).total_seconds()
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_utc')] = df_acc_eva['s_utc'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_utc')] = df_acc_eva['e_utc'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('event_acc')] = df_acc_eva['event_acc'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_spd')] = df_acc_eva['s_spd'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_spd')] = df_acc_eva['e_spd'][i]        
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_crs')] = df_acc_eva['s_crs'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_crs')] = df_acc_eva['e_crs'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_lat')] = df_acc_eva['s_lat'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_lat')] = df_acc_eva['e_lat'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_long')] = df_acc_eva['s_long'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_long')] = df_acc_eva['e_long'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('s_alt')] = df_acc_eva['s_alt'][i]
        df_summary.iloc[rec_num, df_summary.columns.get_loc('e_alt')] = df_acc_eva['e_alt'][i] 
        rec_num += 1

    df_summary.drop_duplicates()
    df_summary = df_summary.sort_values('s_utc', ascending=True) 
    df_summary = df_summary.reset_index(drop=True)

    return df_summary

