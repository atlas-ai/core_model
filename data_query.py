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
        sec_code[i-1] = int(lfc_lt_code + \
        str(severity_bins(rec['sec'+str(i)+'_lat_rt_z'], l1_thr, l2_thr, l3_thr, l4_thr, 'pos')))
        
    if (rec['type']=='lcr') or (rec['type']=='lcl'):
        sec_code[1] = int(9999999)     
    return sec_code


def evt_metrics(sec_code):
    """ 3-dimentional metrics for summary of events
    
    :param sec_code: section code

    :return anti: anticipation, which measures the appropriate speed and acceleration
    :return comf: comfort, which measures the intensity of acceleration and lateral force
    :return cont: control, which measures the direction of lateral force w.r.t events
    """
    evt = int(str(sec_code[0])[0])
    anti = 45.
    comf = 45.
    cont = 45.
    if evt==1:
        for i in range(3):
            if int(str(sec_code[i])[2])>=4:
                anti = anti - int(str(sec_code[i])[2])
            if int(str(sec_code[i])[4])>=3:
                anti = anti - int(str(sec_code[i])[4])
            if int(str(sec_code[i])[4])>=3:
                comf = comf - int(str(sec_code[i])[4])
            if int(str(sec_code[i])[6])>=3:
                comf = comf - int(str(sec_code[i])[6])   
            if int(str(sec_code[i])[5])>1:
                cont = cont - int(str(sec_code[i])[5])
            if int(str(sec_code[i])[2])<=2:
                if int(str(sec_code[i])[6])>=3:
                    cont = cont - int(str(sec_code[i])[6])  
    elif (evt==2) or (evt==3):
        for i in range(3):
            if int(str(sec_code[i])[2])>=4:
                anti = anti - int(str(sec_code[i])[2])
            if int(str(sec_code[i])[4])>=3:
                anti = anti - int(str(sec_code[i])[4])
            if int(str(sec_code[i])[4])>=3:
                comf = comf - int(str(sec_code[i])[4])
            if int(str(sec_code[i])[5])>=3:
                comf = comf - int(str(sec_code[i])[5])   
            if int(str(sec_code[i])[6])>1:
                cont = cont - int(str(sec_code[i])[6])
            if int(str(sec_code[i])[2])<=2:
                if int(str(sec_code[i])[5])>=3:
                    cont = cont - int(str(sec_code[i])[5])  
    elif evt==4:
        if int(str(sec_code[0])[4])>=3:
            anti = anti - int(str(sec_code[0])[4])*2
        if int(str(sec_code[0])[4])>=3:
            comf = comf - int(str(sec_code[0])[4])
        if int(str(sec_code[0])[6])>=3:
            comf = comf - int(str(sec_code[0])[6])   
        if int(str(sec_code[0])[5])>1:
            cont = cont - int(str(sec_code[0])[5])
        if int(str(sec_code[0])[2])<=2:
            if int(str(sec_code[0])[6])>=3:
                cont = cont - int(str(sec_code[0])[6]) 
        if int(str(sec_code[2])[4])>=3:
            anti = anti - int(str(sec_code[0])[4])
        if int(str(sec_code[2])[4])>=3:
            comf = comf - int(str(sec_code[1])[4])
        if int(str(sec_code[2])[5])>=3:
            comf = comf - int(str(sec_code[1])[5])   
        if int(str(sec_code[2])[6])>1:
            cont = cont - int(str(sec_code[1])[6])
        if int(str(sec_code[2])[2])<=2:
            if int(str(sec_code[1])[5])>=3:
                cont = cont - int(str(sec_code[1])[5])        
    elif evt==5:
        if int(str(sec_code[0])[4])>=3:
            anti = anti - int(str(sec_code[0])[4])*2
        if int(str(sec_code[0])[4])>=3:
            comf = comf - int(str(sec_code[0])[4])
        if int(str(sec_code[0])[5])>=3:
            comf = comf - int(str(sec_code[0])[5])   
        if int(str(sec_code[0])[6])>1:
            cont = cont - int(str(sec_code[0])[6])
        if int(str(sec_code[0])[2])<=2:
            if int(str(sec_code[0])[5])>=3:
                cont = cont - int(str(sec_code[0])[5]) 
        if int(str(sec_code[2])[4])>=3:
            anti = anti - int(str(sec_code[0])[4])
        if int(str(sec_code[2])[4])>=3:
            comf = comf - int(str(sec_code[1])[4])
        if int(str(sec_code[2])[6])>=3:
            comf = comf - int(str(sec_code[1])[6])   
        if int(str(sec_code[2])[5])>1:
            cont = cont - int(str(sec_code[1])[5])
        if int(str(sec_code[2])[2])<=2:
            if int(str(sec_code[1])[6])>=3:
                cont = cont - int(str(sec_code[1])[6])         
    anti = round(anti/45*5,1)
    comf = round(comf/45*5,1)
    cont = round(cont/45*5,1)    
    return anti, comf, cont


def display_track_info(df, code_sys, l1_thr, l2_thr, l3_thr, l4_thr, acc_fac):
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
    df_display['anticipation']=np.NaN
    df_display['comfort']=np.NaN
    df_display['control']=np.NaN
    df_display['sec1_code']=np.NaN
    df_display['sec2_code']=np.NaN
    df_display['sec3_code']=np.NaN
    df_display['sec1_desc']=np.NaN
    df_display['sec2_desc']=np.NaN
    df_display['sec3_desc']=np.NaN
    df_display['sec1_diag']=np.NaN
    df_display['sec2_diag']=np.NaN
    df_display['sec3_diag']=np.NaN
    df_display['sec1_pct']=np.NaN
    df_display['sec2_pct']=np.NaN
    df_display['sec3_pct']=np.NaN
    dt_len = df_display.shape[0]
    for i in range(dt_len):
        if (df_display['type'][i]=='rtt') or (df_display['type'][i]=='ltt') or (df_display['type'][i]=='utn') or\
        (df_display['type'][i]=='lcr')  or (df_display['type'][i]=='lcl'):
            sec_code = evt_sec_codes(df_display.iloc[i], l1_thr, l2_thr, l3_thr, l4_thr)
            anti, comf,cont = evt_metrics(sec_code)
            df_display.iloc[i, df_display.columns.get_loc('sec1_code')] = sec_code[0]
            df_display.iloc[i, df_display.columns.get_loc('sec2_code')] = sec_code[1]
            df_display.iloc[i, df_display.columns.get_loc('sec3_code')] = sec_code[2]
            sec1_idx = code_sys.index.searchsorted(sec_code[0])
            sec2_idx = code_sys.index.searchsorted(sec_code[1])
            sec3_idx = code_sys.index.searchsorted(sec_code[2])
            df_display.iloc[i, df_display.columns.get_loc('sec1_desc')] = code_sys['description_chn'].iloc[sec1_idx]
            df_display.iloc[i, df_display.columns.get_loc('sec2_desc')] = code_sys['description_chn'].iloc[sec2_idx]
            df_display.iloc[i, df_display.columns.get_loc('sec3_desc')] = code_sys['description_chn'].iloc[sec3_idx]            
            df_display.iloc[i, df_display.columns.get_loc('sec1_diag')] = code_sys['diagnosis_chn'].iloc[sec1_idx]
            df_display.iloc[i, df_display.columns.get_loc('sec2_diag')] = code_sys['diagnosis_chn'].iloc[sec2_idx]
            df_display.iloc[i, df_display.columns.get_loc('sec3_diag')] = code_sys['diagnosis_chn'].iloc[sec3_idx] 
            df_display.iloc[i, df_display.columns.get_loc('sec1_pct')] = code_sys['pct_of_occurrence'].iloc[sec1_idx]
            df_display.iloc[i, df_display.columns.get_loc('sec2_pct')] = code_sys['pct_of_occurrence'].iloc[sec2_idx]
            df_display.iloc[i, df_display.columns.get_loc('sec3_pct')] = code_sys['pct_of_occurrence'].iloc[sec3_idx]
            df_display.iloc[i, df_display.columns.get_loc('anticipation')] = anti
            df_display.iloc[i, df_display.columns.get_loc('comfort')] = comf
            df_display.iloc[i, df_display.columns.get_loc('control')] = cont
        elif df_display['type'][i]=='exa':
            df_display.iloc[i, df_display.columns.get_loc('sec1_code')] = \
            int(str(1) + str(int(df_display['sec1_spd_bin'][i])) + str(severity_bins(df_display['sec1_acc_z'][i], l1_thr, l2_thr, l3_thr, l4_thr, 'pos', acc_fac)))
            df_display.iloc[i, df_display.columns.get_loc('sec2_code')] = 999
            df_display.iloc[i, df_display.columns.get_loc('sec3_code')] = 999
        elif df_display['type'][i]=='exd':
            df_display.iloc[i, df_display.columns.get_loc('sec1_code')] = \
            int(str(2) + str(int(df_display['sec1_spd_bin'][i])) + str(severity_bins(df_display['sec1_dec_z'][i], l1_thr, l2_thr, l3_thr, l4_thr, 'neg', acc_fac)))         
            df_display.iloc[i, df_display.columns.get_loc('sec2_code')] = 999
            df_display.iloc[i, df_display.columns.get_loc('sec3_code')] = 999
    
    df_display = df_display[(df_display['type']=='rtt') | (df_display['type']=='ltt') |(df_display['type']=='utn') |\
                            (df_display['type']=='lcr') | (df_display['type']=='lcl') |\
                            (((df_display['type']=='exa') | (df_display['type']=='exd')) & (df_display['score']!=100.))]
    df_display = df_display.reset_index(drop=True)    
    df_display = df_display[['uuid','type','prob','score','d','s_utc','e_utc','event_acc','s_spd','e_spd','s_crs','e_crs',\
                             's_lat','e_lat','s_long','e_long','s_alt','e_alt','anticipation','comfort','control',\
                             'sec1_s_spd','sec1_e_spd','sec1_code','sec1_desc','sec1_diag','sec1_pct',\
                             'sec2_s_spd','sec2_e_spd','sec2_code','sec2_desc','sec2_diag','sec2_pct',\
                             'sec3_s_spd','sec3_e_spd','sec3_code','sec3_desc','sec3_diag','sec3_pct',\
                             'acc_1','acc_2','acc_3','acc_4','acc_5','acc_6','acc_7','acc_8','acc_9','acc_10',\
                             'acc_11','acc_12','acc_13','acc_14','acc_15','acc_16','acc_17','acc_18','acc_19','acc_20',\
                             'lfc_1','lfc_2','lfc_3','lfc_4','lfc_5','lfc_6','lfc_7','lfc_8','lfc_9','lfc_10',\
                             'lfc_11','lfc_12','lfc_13','lfc_14','lfc_15','lfc_16','lfc_17','lfc_18','lfc_19','lfc_20',\
                             'spd_1','spd_2','spd_3','spd_4','spd_5','spd_6','spd_7','spd_8','spd_9','spd_10',\
                             'spd_11','spd_12','spd_13','spd_14','spd_15','spd_16','spd_17','spd_18','spd_19','spd_20']]       
    return df_display


def focus_algo(df_sum, df_display):
    """focus measures the attention that drivers pay while driving
    
    :param df_sum: cleaned results
    :param df_display: displayed results
    :return: score for focus
    """
    exd_num = df_sum[df_sum['type']=='exd'].shape[0]
    focus_num = df_display[df_display['type']=='exd'].shape[0]
    if focus_num!=0:
        tot_score = 100. - exd_num/focus_num*100.
    else:
        tot_score = 100.
    tot_score = round(tot_score/20,1)
    return tot_score


def efficiency_algo(df_sum):
    """efficiency measures the intensity of acceleration
    
    :param df_sum: cleaned results
    :return: score for efficiency
    """ 
    tot_score=100.
    for i in range(3):
        acc_score = df_sum['sec'+str(i+1)+'_acc_z'].where(df_sum['sec'+str(i+1)+'_acc_z']>3).sum()\
        /(df_sum['sec'+str(i+1)+'_acc_z'].count()*3)
        dec_score = df_sum['sec'+str(i+1)+'_dec_z'].where(df_sum['sec'+str(i+1)+'_dec_z']<-3).sum()\
        /(df_sum['sec'+str(i+1)+'_dec_z'].count()*-3)
        if acc_score>0: 
            tot_score = tot_score - acc_score*50. 
        if dec_score>0:   
            tot_score = tot_score - dec_score*50.  
    tot_score = round(tot_score/20,1)
    return tot_score


def anticipation_algo(df_sum):
    """anticipation measures whether drivers have tailgating behaviour 
        
    :param df_sum: cleaned results
    :return: score for anticipation
    """
    df_tg = df_sum[(df_sum['type']=='exa')|(df_sum['type']=='exd')].copy(deep=True)
    df_tg.reset_index(inplace=True)
    dtlen = df_tg.shape[0]
    tg_score = (dtlen-1)*20
    tg_label = np.zeros(dtlen)
    tg_dt = df_tg['s_utc'].diff().bfill()/np.timedelta64(1, 's')
    for i in range(1, dtlen):
        if tg_dt[i]<=15.:
            if (df_tg['type'][i]=='exa') and (df_tg['type'][i-1]=='exd'):
                tg_label[i] = 20
            elif (df_tg['type'][i]=='exd') and (df_tg['type'][i-1]=='exa'):
                tg_label[i] = 20
            elif (df_tg['type'][i]=='exd') and (df_tg['type'][i-1]=='exd'):
                tg_label[i] = 15
            elif (df_tg['type'][i]=='exa') and (df_tg['type'][i-1]=='exa'):
                tg_label[i] = 10
    tot_score = 100 - tg_label.sum()/tg_score*100
    tot_score = round(tot_score/20,1)
    return tot_score


def control_algo(df_display):
    """control measures whether drivers have full control of the vehicles
    
    :param df_display: displayed results
    :return: score for control
    """
    tot_score = (df_display['control'].mean()+ df_display['comfort'].mean())*10
    tot_score = round(tot_score/20,1)
    return tot_score


def legality_algo():
    """legality measures breaches of traffic laws
    
    dummy algo for now, place holder
    """
    tot_score = 100
    tot_score = round(tot_score/20,1)
    return tot_score


def track_info_summary(df_sum, df_display):
    """ summarise track results
    
    :param df_sum: results from cleaned db
    :param df_display: results from displayed db
    :return: track summary table in one line
    """    
    res = pd.DataFrame(np.nan, index=np.arange(1), columns=['uuid','s_utc','e_utc','dur','dist',\
                       'ave_spd','no_rtt','no_ltt','no_utn','no_lcr','no_lcl','no_exa','no_exd',\
                       'rtt_score','ltt_score','utn_score','lcr_score','lcl_score','exa_score','exd_score',\
                       'performance','focus','efficiency','anticipation', 'control','legality'])
    res.iloc[0, res.columns.get_loc('uuid')] = df_sum['uuid'][0]
    res.iloc[0, res.columns.get_loc('no_rtt')] = df_display['type'].where(df_display['type']=='rtt').count()
    res.iloc[0, res.columns.get_loc('rtt_score')] = round(df_display['score'].where(df_display['type']=='rtt').mean()/20,1)
    res.iloc[0, res.columns.get_loc('no_ltt')] = df_display['type'].where(df_display['type']=='ltt').count()
    res.iloc[0, res.columns.get_loc('ltt_score')] = round(df_display['score'].where(df_display['type']=='ltt').mean()/20,1)
    res.iloc[0, res.columns.get_loc('no_utn')] = df_display['type'].where(df_display['type']=='utn').count()
    res.iloc[0, res.columns.get_loc('utn_score')] = round(df_display['score'].where(df_display['type']=='utn').mean()/20,1)
    res.iloc[0, res.columns.get_loc('no_lcr')] = df_display['type'].where(df_display['type']=='lcr').count()
    res.iloc[0, res.columns.get_loc('lcr_score')] = round(df_display['score'].where(df_display['type']=='lcr').mean()/20,1)
    res.iloc[0, res.columns.get_loc('no_lcl')] = df_display['type'].where(df_display['type']=='lcl').count()
    res.iloc[0, res.columns.get_loc('lcl_score')] = round(df_display['score'].where(df_display['type']=='lcl').mean()/20,1)
    res.iloc[0, res.columns.get_loc('no_exa')] = df_display['type'].where(df_display['type']=='exa').count()
    res.iloc[0, res.columns.get_loc('exa_score')] = round(df_display['score'].where(df_display['type']=='exa').mean()/20,1)
    res.iloc[0, res.columns.get_loc('no_exd')] = df_display['type'].where(df_display['type']=='exd').count()
    res.iloc[0, res.columns.get_loc('exd_score')] = round(df_display['score'].where(df_display['type']=='exd').mean()/20,1)
    res.iloc[0, res.columns.get_loc('focus')] = focus_algo(df_sum, df_display)
    res.iloc[0, res.columns.get_loc('efficiency')] = efficiency_algo(df_sum)
    res.iloc[0, res.columns.get_loc('anticipation')] = anticipation_algo(df_sum)
    res.iloc[0, res.columns.get_loc('control')] = control_algo(df_display)
    res.iloc[0, res.columns.get_loc('legality')] = legality_algo()
    overall_performance = round((df_display['score'].mean()/20 + res['focus'][0] + res['efficiency'][0] \
                                 + res['anticipation'][0] + res['control'][0] + res['legality'][0])/6., 1)
    res.iloc[0, res.columns.get_loc('performance')] = overall_performance
    
    return res


