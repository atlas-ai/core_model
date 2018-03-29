#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 14:45:33 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import distributions as dst


def eva_resampling(df_evt, df_acc, acc_x, acc_x_gps, acc_y, rot_z, spd, crs, samp_rate, g=9.8):
    """ resample detection results for evaluation process
    
    :param df_evt: detected events
    :param df_acc: detected excess acceleration
    :param acc_x: acceleration from IMU (G)
    :param acc_x_gps: acceleration derived from GPS speed (G)
    :param acc_y: lateral force from IMU (G)
    :param spd: speed from GPS in km/hr    
    :param samp_rate: sampling rate
    :return : resampled data for evaluation
    """
    df_res = pd.concat([df_evt, df_acc], axis=0)
    df_res = df_res.sort_values('s_utc', ascending=True) 
    df_res = df_res.reset_index(drop=True)
    
    fac = 0.5   
    acc_e = np.zeros(20)
    spd_e = np.zeros(20)    
    t = np.zeros(20)
    acc_x_res = np.zeros(20)
    acc_x_gps_res = np.zeros(20)
    spd_res = np.zeros(20)
    acc_y_res = np.zeros(20)
    crs_res = np.zeros(20)
    rot_res = np.zeros(20)
            
    for i in range (20): 
        df_res['spd_'+str(i+1)]=np.NaN
        df_res['acc_'+str(i+1)]=np.NaN
        df_res['lfc_'+str(i+1)]=np.NaN
        df_res['crs_'+str(i+1)]=np.NaN
        df_res['rot_'+str(i+1)]=np.NaN
        
    dt_len = df_res.shape[0] 
    for i in range(dt_len):        
        beg_idx = acc_x.index.searchsorted(df_res['s_utc'][i]) 
        end_idx = acc_x.index.searchsorted(df_res['e_utc'][i])
        stepSize = (end_idx-beg_idx+1)/20
        
        for j in range (20):
            t[j] = acc_x.index[int(np.floor(beg_idx+j*stepSize))].value
            acc_x_res[j] = acc_x[int(np.floor(beg_idx+j*stepSize))]
            acc_x_gps_res[j] = acc_x_gps[int(np.floor(beg_idx+j*stepSize))]
            spd_res[j] = spd[int(np.floor(beg_idx+j*stepSize))]
            acc_y_res[j] = acc_y[int(np.floor(beg_idx+j*stepSize))]
            crs_res[j] = crs[int(np.floor(beg_idx+j*stepSize))]
            rot_res[j] = rot_z[int(np.floor(beg_idx+j*stepSize))]
            
        delta_t = pd.Series(t).diff().bfill()/1000000000        
        acc_e = (1-fac)*acc_x_res + fac*acc_x_gps_res
        spd_e = (acc_e*g*delta_t).cumsum()*3.6 + df_res['s_spd'][i]  
            
        for j in range(20):
            df_res.iloc[i, df_res.columns.get_loc('spd_'+str(j+1))] = spd_e[j]
            df_res.iloc[i, df_res.columns.get_loc('acc_'+str(j+1))] = acc_e[j]
            df_res.iloc[i, df_res.columns.get_loc('lfc_'+str(j+1))] = acc_y_res[j]
            df_res.iloc[i, df_res.columns.get_loc('crs_'+str(j+1))] = crs_res[j]
            df_res.iloc[i, df_res.columns.get_loc('rot_'+str(j+1))] = rot_res[j]
 
    return df_res


def evt_sec_zscore(df_rec, param_rtt, param_ltt, param_utn, param_lcr, param_lcl, evt_type, samp_rate):
    """ evaluation model for a single event 
        
    :param df_res: series of detected events
    :param param_rtt: coefficients for right turns evaluation
    :param param_ltt: coefficients for left turns evaluation
    :param param_utn: coefficients for u-turns evaluation
    :param param_lcr: coefficients for lane changes to the right evaluation
    :param param_lcl: coefficients for lane changes to the left evaluation
    :param evt_type: type of event (right turn, left turn, lane change to right, lane change to left)
    :param samp_rate: sampling rate of data collection
    :return: speed bin, z score of acceleration, z score of deceleration, 
    z score of lateral force when making left turn, z score of lateral force when making a right turn
    for three sections (entering, middle, exiting)
    """        
    acc_col = ['acc_1','acc_2','acc_3','acc_4','acc_5','acc_6','acc_7','acc_8','acc_9','acc_10',\
               'acc_11','acc_12','acc_13','acc_14','acc_15','acc_16','acc_17','acc_18','acc_19','acc_20']
    lfc_col = ['lfc_1','lfc_2','lfc_3','lfc_4','lfc_5','lfc_6','lfc_7','lfc_8','lfc_9','lfc_10',\
               'lfc_11','lfc_12','lfc_13','lfc_14','lfc_15','lfc_16','lfc_17','lfc_18','lfc_19','lfc_20']

    if (evt_type=='rtt') or (evt_type=='ltt') or (evt_type=='utn'):
        if evt_type=='rtt':
            coef = param_rtt
        elif evt_type=='ltt':
            coef = param_ltt       
        elif evt_type=='utn':
            coef = param_utn        
        dt_num = 3
        sec_idx = [0,5,14,20]
    elif (evt_type=='lcr') or (evt_type=='lcl'):
        if evt_type=='lcr':
            coef = param_lcr
        elif evt_type=='lcl':
            coef = param_lcl   
        dt_num = 2
        sec_idx = [0,10,20]

    sec_s_spd = np.zeros(dt_num)
    sec_e_spd = np.zeros(dt_num)
    sec_spd = np.zeros(dt_num)
    spd_bin = np.zeros(dt_num)
    acc_z = np.zeros(dt_num)
    dec_z = np.zeros(dt_num)
    lfc_lt_z = np.zeros(dt_num)
    lfc_rt_z = np.zeros(dt_num)
    
    for i in range(dt_num):
        
        #average speed
        sec_s_spd[i] = df_rec['spd_'+str(sec_idx[i]+1)]
        sec_e_spd[i] = df_rec['spd_'+str(sec_idx[i+1])]
        sec_spd[i] = (sec_s_spd[i] + sec_e_spd[i])/2
        
        sec_acc_col=[]
        sec_lfc_col=[]        
        for j in range(sec_idx[i], sec_idx[i+1]):
            sec_acc_col.append(acc_col[j])
            sec_lfc_col.append(lfc_col[j])
        
        #speed bin subject to average speed
        spd_bin[i] = dst.spd_bins(sec_spd[i])        
        #z score for acceleration (max)       
        acc_z[i] = dst.z_score(df_rec[sec_acc_col].where(df_rec[sec_acc_col]>=0).max(),\
                      coef[evt_type+'_sec'+str(i+1)+'_acc_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_type+'_sec'+str(i+1)+'_acc_var'].iloc[(spd_bin[i]-1).astype(int)]))
        #z score for deceleration (min)       
        dec_z[i] = dst.z_score(df_rec[sec_acc_col].where(df_rec[sec_acc_col]<0).min(),\
                      coef[evt_type+'_sec'+str(i+1)+'_dec_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_type+'_sec'+str(i+1)+'_dec_var'].iloc[(spd_bin[i]-1).astype(int)]))
        #z score for lateral force (left turn, tilting to the right) (min)       
        lfc_lt_z[i] = dst.z_score(df_rec[sec_lfc_col].where(df_rec[sec_lfc_col]<0).min(),\
                      coef[evt_type+'_sec'+str(i+1)+'_lfc_lt_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_type+'_sec'+str(i+1)+'_lfc_lt_var'].iloc[(spd_bin[i]-1).astype(int)]))
        #z score for lateral force (right turn, tilting to the left) (max)       
        lfc_rt_z[i] = dst.z_score(df_rec[sec_lfc_col].where(df_rec[sec_lfc_col]>=0).max(),\
                      coef[evt_type+'_sec'+str(i+1)+'_lfc_rt_ave'].iloc[(spd_bin[i]-1).astype(int)],\
                      np.sqrt(coef[evt_type+'_sec'+str(i+1)+'_lfc_rt_var'].iloc[(spd_bin[i]-1).astype(int)]))
            
    return sec_s_spd, sec_e_spd, spd_bin, acc_z, dec_z, lfc_lt_z, lfc_rt_z


def evt_score_algo(acc_z, dec_z, lfc_lt_z, lfc_rt_z, l1_thr, l2_thr, l3_thr, l4_thr):
    """ calculate score for individual events
    
    :param acc_z: z score for acceleration
    :param dec_z: z score for deceleration
    :param lfc_lt_z: z score for lateral force when makeing a left turn (tilting to right)
    :param lfc_rt_z: z score for lateral force when makeing a right turn (tilting to left)
    :param l1_thr: l1 threshold for severity measurement
    :param l2_thr: l2 threshold for severity measurement
    :param l3_thr: l3 threshold for severity measurement
    :param l4_thr: l4 threshold for severity measurement
    :return : total score of the event
    """    
    l1_score = 10
    l2_score = 20
    l3_score = 30
    l4_score = 40
    
    acc_l1_score = np.where((dst.compare_nan_array(np.greater_equal, acc_z, l1_thr)) &\
                            (dst.compare_nan_array(np.less, acc_z, l2_thr)),\
                            (acc_z-l1_thr) * l1_score, 0)
    acc_l2_score = np.where((dst.compare_nan_array(np.greater_equal, acc_z, l2_thr)) &\
                            (dst.compare_nan_array(np.less, acc_z, l3_thr)),\
                            (acc_z-l2_thr) * l2_score + l1_score, 0)
    acc_l3_score = np.where((dst.compare_nan_array(np.greater_equal, acc_z, l3_thr)) &\
                            (dst.compare_nan_array(np.less, acc_z, l3_thr)),\
                            (acc_z-l3_thr) * l3_score + l2_score, 0)
    acc_l4_score = np.where((dst.compare_nan_array(np.greater_equal, acc_z, l4_thr)),\
                            (acc_z-l4_thr) * l4_score + l3_score, 0)
    
    dec_l1_score = np.where((dst.compare_nan_array(np.less_equal, dec_z, -1.*l1_thr)) &\
                            (dst.compare_nan_array(np.greater, dec_z, -1.*l2_thr)),\
                            (-1.*(dec_z+l1_thr)) * l1_score, 0)  
    dec_l2_score = np.where((dst.compare_nan_array(np.less_equal, dec_z, -1.*l2_thr)) &\
                            (dst.compare_nan_array(np.greater, dec_z, -1.*l3_thr)),\
                            (-1.*(dec_z+l2_thr)) * l2_score + l1_score, 0)  
    dec_l3_score = np.where((dst.compare_nan_array(np.less_equal, dec_z, -1.*l3_thr)) &\
                            (dst.compare_nan_array(np.greater, dec_z, -1.*l4_thr)),\
                            (-1.*(dec_z+l3_thr)) * l3_score + l2_score, 0)  
    dec_l4_score = np.where((dst.compare_nan_array(np.less_equal, dec_z, -1.*l4_thr)),\
                            (-1.*(dec_z+l4_thr)) * l4_score + l3_score, 0)    

    lfc_lt_l1_score = np.where((dst.compare_nan_array(np.less_equal, lfc_lt_z, -1.*l1_thr)) &\
                               (dst.compare_nan_array(np.greater, lfc_lt_z, -1.*l2_thr)),\
                               (-1.*(lfc_lt_z+l1_thr)) * l1_score, 0)  
    lfc_lt_l2_score = np.where((dst.compare_nan_array(np.less_equal, lfc_lt_z, -1.*l2_thr)) &\
                               (dst.compare_nan_array(np.greater, lfc_lt_z, -1.*l3_thr)),\
                               (-1.*(lfc_lt_z+l2_thr)) * l2_score + l1_score, 0)  
    lfc_lt_l3_score = np.where((dst.compare_nan_array(np.less_equal, lfc_lt_z, -1.*l3_thr)) &\
                               (dst.compare_nan_array(np.greater, lfc_lt_z, -1.*l4_thr)),\
                               (-1.*(lfc_lt_z+l3_thr)) * l3_score + l2_score, 0)  
    lfc_lt_l4_score = np.where((dst.compare_nan_array(np.less_equal, lfc_lt_z, -1.*l4_thr)),\
                               (-1.*(lfc_lt_z+l4_thr)) * l4_score + l3_score, 0)                         

    lfc_rt_l1_score = np.where((dst.compare_nan_array(np.greater_equal, lfc_rt_z, l1_thr)) &\
                               (dst.compare_nan_array(np.less, lfc_rt_z, l2_thr)),\
                               (lfc_rt_z-l1_thr) * l1_score, 0)
    lfc_rt_l2_score = np.where((dst.compare_nan_array(np.greater_equal, lfc_rt_z, l2_thr)) &\
                               (dst.compare_nan_array(np.less, lfc_rt_z, l3_thr)),\
                               (lfc_rt_z-l2_thr) * l2_score + l1_score, 0)
    lfc_rt_l3_score = np.where((dst.compare_nan_array(np.greater_equal, lfc_rt_z, l3_thr)) &\
                               (dst.compare_nan_array(np.less, lfc_rt_z, l3_thr)),\
                               (lfc_rt_z-l3_thr) * l3_score + l2_score, 0)
    lfc_rt_l4_score = np.where((dst.compare_nan_array(np.greater_equal, lfc_rt_z, l4_thr)),\
                               (lfc_rt_z-l4_thr) * l4_score + l3_score, 0)

    score = round((acc_l1_score + acc_l2_score + acc_l3_score + acc_l4_score + \
                   dec_l1_score + dec_l2_score + dec_l3_score + dec_l4_score + \
                   lfc_lt_l1_score + lfc_lt_l2_score + lfc_lt_l3_score + lfc_lt_l4_score + \
                   lfc_rt_l1_score + lfc_rt_l2_score + lfc_rt_l3_score + lfc_rt_l4_score).sum(),0)
    
    if score < 100.0:
        tot_score  = 100.0 - score
    else:
        tot_score = 0.

    return tot_score
    

def ex_acc_score_algo(df_rec, param_acc, acc_type, g=9.8):
    """ score individual excess acceleration 
    
    :param df_res: series of detected acceleration
    :param param_acc: coefficients for excess acceleration
    :param acc_type: type of acceleration or deceleration
    :param acc_thr: threshold of z-score that acceleration breaches
    :return : evaluation results for excess acceleration
    """ 
    exc_score = 10
    acc_col = ['acc_1','acc_2','acc_3','acc_4','acc_5','acc_6','acc_7','acc_8','acc_9','acc_10',\
               'acc_11','acc_12','acc_13','acc_14','acc_15','acc_16','acc_17','acc_18','acc_19','acc_20']
    acc = df_rec[acc_col]
    
    if acc_type=='exa':
        acc_max = acc.max()
        maxacc_col = acc.idxmax()
        maxacc_s_spd = maxacc_col.replace('acc','spd')
        s_spd = df_rec[maxacc_s_spd]
        if maxacc_s_spd=='spd_20':
            maxacc_e_spd = maxacc_s_spd
        else:
            maxacc_e_spd = 'spd_'+ str(int(maxacc_s_spd.split('_')[1])+1)
        e_spd = df_rec[maxacc_e_spd]
        spd_bin = dst.spd_bins(s_spd)
        acc_zscore = dst.z_score(acc_max,param_acc['acc_ave'][int(spd_bin-1)],np.sqrt(param_acc['acc_var'][int(spd_bin-1)])) 
        if acc_zscore<=0:
            tot_score = 100.
        else:
            if acc_zscore*exc_score < 100.0:
                tot_score = 100.0 - round(acc_zscore*exc_score, 0)
            else:
                tot_score = 0.
    elif acc_type=='exd':
        dec_max = acc.min()
        maxdec_col = acc.idxmin()
        maxdec_s_spd = maxdec_col.replace('acc','spd')
        s_spd = df_rec[maxdec_s_spd]
        if maxdec_s_spd=='spd_20':
            maxdec_e_spd = maxdec_s_spd
        else: 
            maxdec_e_spd = 'spd_'+ str(int(maxdec_s_spd.split('_')[1])+1)
        e_spd = df_rec[maxdec_e_spd]
        spd_bin = dst.spd_bins(s_spd)
        acc_zscore = dst.z_score(dec_max,param_acc['dec_ave'][int(spd_bin-1)],np.sqrt(param_acc['dec_var'][int(spd_bin-1)])) 
        if acc_zscore>=0:
            tot_score = 100.
        else:
            if -acc_zscore*exc_score < 100.0:
                tot_score = 100.0 - round(-acc_zscore*exc_score, 0)
            else:
                tot_score = 0.
    return tot_score, acc_zscore, spd_bin, s_spd, e_spd


def evt_n_acc_evaluation(df_res, param_rtt, param_ltt, param_utn, param_lcr, param_lcl, param_acc,\
                         samp_rate, l1_thr, l2_thr, l3_thr, l4_thr, track_id):
    """ evaluation model for events and accelerations 
        
    :param df_res: detection results
    :param param_rtt: coefficients for right turns evaluation
    :param param_ltt: coefficients for left turns evaluation
    :param param_utn: coefficients for u-turns evaluation
    :param param_lcr: coefficients for lane changes to the right evaluation
    :param param_lcl: coefficients for lane changes to the left evaluation
    :param param_acc: coefficients for excess acceleration
    :param samp_rate: sampling rate of data collection
    :param l1_thr: l1 threshold for severity measurement
    :param l2_thr: l2 threshold for severity measurement
    :param l3_thr: l3 threshold for severity measurement
    :param l4_thr: l4 threshold for severity measurement
    :param track_id: uuid
    :return: evaluation result database
    """    
    df_eva = df_res.copy(deep=True)
    df_eva['uuid'] = track_id
    df_eva['score'] = np.NaN
    for i in range (3): 
        df_eva['sec'+str(i+1)+'_s_spd']=np.NaN
        df_eva['sec'+str(i+1)+'_e_spd']=np.NaN
        df_eva['sec'+str(i+1)+'_spd_bin']=np.NaN
        df_eva['sec'+str(i+1)+'_acc_z']=np.NaN
        df_eva['sec'+str(i+1)+'_dec_z']=np.NaN
        df_eva['sec'+str(i+1)+'_lat_lt_z']=np.NaN
        df_eva['sec'+str(i+1)+'_lat_rt_z']=np.NaN
        
    dt_len = df_eva.shape[0]
    if dt_len!=0:
        for i in range(dt_len):        
            evt_type = df_eva['type'][i]
            if (evt_type=='rtt') or (evt_type=='ltt') or (evt_type=='utn') or (evt_type=='lcr') or (evt_type=='lcl'):            
                sec_s_spd, sec_e_spd, spd_bin, acc_z, dec_z, lfc_lt_z, lfc_rt_z = \
                evt_sec_zscore(df_res.iloc[i], param_rtt, param_ltt, param_utn, param_lcr, param_lcl, evt_type, samp_rate)            
                tot_score = evt_score_algo(acc_z, dec_z, lfc_lt_z, lfc_rt_z, l1_thr, l2_thr, l3_thr, l4_thr)
                df_eva.iloc[i, df_eva.columns.get_loc('score')] = tot_score
            
                if (evt_type=='rtt') or (evt_type=='ltt') or (evt_type=='utn'):
                    dt_idx = [0, 1, 2]
                    sec_idx = [1, 2, 3]
                elif (evt_type=='lcr') or (evt_type=='lcl'):
                    dt_idx = [0, 1]
                    sec_idx = [1, 3]
                for j in dt_idx:
                    df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_s_spd')] = sec_s_spd[j]
                    df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_e_spd')] = sec_e_spd[j]
                    df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_spd_bin')] = spd_bin[j]
                    if sec_e_spd[j]>=sec_s_spd[j]:
                        df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_acc_z')] = acc_z[j]
                    else:
                        df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_dec_z')] = dec_z[j]
                        df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_lat_lt_z')] = lfc_lt_z[j]
                        df_eva.iloc[i, df_eva.columns.get_loc('sec'+str(sec_idx[j])+'_lat_rt_z')] = lfc_rt_z[j] 
            elif (evt_type=='exa') or (evt_type=='exd'):
                tot_score, acc_zscore, spd_bin, s_spd, e_spd = ex_acc_score_algo(df_res.iloc[i], param_acc, evt_type)
                df_eva.iloc[i, df_eva.columns.get_loc('sec1_s_spd')] = s_spd
                df_eva.iloc[i, df_eva.columns.get_loc('sec1_e_spd')] = e_spd
                df_eva.iloc[i, df_eva.columns.get_loc('sec1_spd_bin')] = spd_bin
                df_eva.iloc[i, df_eva.columns.get_loc('score')] = tot_score
                if evt_type=='exa':
                    df_eva.iloc[i, df_eva.columns.get_loc('sec1_acc_z')] = acc_zscore
                else:
                    df_eva.iloc[i, df_eva.columns.get_loc('sec1_dec_z')] = acc_zscore
                        
    df_eva = df_eva[['uuid','type','prob','score','d','s_utc','e_utc',\
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
    
    return df_eva
    
