# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:51:27 2017

@author: seanx
"""

import pandas as pd
import numpy as np
import timeit


def Event_Evaluation(df, df_event):
    start = timeit.default_timer()
    
##############################################################################################################################
#                                                          Initialisation                                                    #
##############################################################################################################################
    
    #Read coefficients
    coef_rtt = pd.read_csv('Evaluation_Coefficients.csv', header=0, index_col=0, nrows=5)
    coef_ltt = pd.read_csv('Evaluation_Coefficients.csv', header=8, index_col=0, nrows=5)
    coef_lcr = pd.read_csv('Evaluation_Coefficients.csv', header=16, index_col=0, nrows=5)
    coef_lcl = pd.read_csv('Evaluation_Coefficients.csv', header=24, index_col=0, nrows=5)
    coef_turns = pd.merge(coef_rtt, coef_ltt, how='left', left_index=True, right_index=True)
    coef_lanes = pd.merge(coef_lcr, coef_lcl, how='left', left_index=True, right_index=True)
    coef = pd.merge(coef_turns, coef_lanes, how='left', left_index=True, right_index=True)
        
    #Function to define speed bins
    def speed_bins(speed):
        if speed<15:
            SpdBin = 0
        elif speed>=15 and speed<25:
            SpdBin = 1
        elif speed>=25 and speed<35:
            SpdBin = 2
        elif speed>=35 and speed<45:
            SpdBin = 3
        elif speed>+45:
            SpdBin = 4 
        return SpdBin
    
    #Function to define z-scores
    def z_Score(val, ave, std):
        return ((val-ave)/std)
    
    dataPoints = 50
    eventLen = df_event.shape[0]
    spd_ave = np.zeros((eventLen,3))
    lonF_max_z = np.zeros((eventLen,3))
    lonF_min_z = np.zeros((eventLen,3))
    latF_max_z = np.zeros((eventLen,3))
    latF_min_z = np.zeros((eventLen,3))
    
    #Loop through Event DataFrame to evaluate individual events
    for i in range(eventLen):
        beg_index = df_event['Start_Index'].iloc[i]
        end_index = df_event['End_Index'].iloc[i]
        eventDuration = end_index - beg_index + 1
        stepSize = (eventDuration-1)/dataPoints
        latForce = np.zeros(eventDuration)
        lonForce = np.zeros(eventDuration)
        spd =  np.zeros(eventDuration)
        for j in range(eventDuration):         
            lonForce[j] = df['user_acc_y(G)'].iloc[beg_index + j]
            latForce[j] = df['user_acc_x(G)'].iloc[beg_index + j]
            spd[j] = df['speed(m/s)'].iloc[beg_index + j]*3.6
               
        spd_bin = np.zeros(3)        
        if df_event['Type'].iloc[i] =='RTT':
            event_type = 'RTT'
            sec_idx = [0,15,40,50] #15+25+10=50 three sections
            sec = 3
        elif df_event['Type'].iloc[i] =='LTT':
            event_type = 'LTT'
            sec_idx = [0,15,40,50]
            sec = 3
        elif df_event['Type'].iloc[i] =='LCR':
            event_type = 'LCR'
            sec_idx = [0,25,50]
            sec = 2 
        elif df_event['Type'].iloc[i] =='LCL':
            event_type = 'LCL'
            sec_idx = [0,25,50]
            sec = 2
        
        #Loop through to calculate z scores depending on speed bins    
        for j in range(sec):
            spd_ave[i,j] = spd[(sec_idx[j]*stepSize).astype(int):(sec_idx[j+1]*stepSize).astype(int)].mean()
            spd_bin[j] = speed_bins(spd_ave[i,j])
            lonF_max_z[i,j] = z_Score(lonForce[(sec_idx[j]*stepSize).astype(int):(sec_idx[j+1]*stepSize).astype(int)].max(),\
                      coef[event_type+'_Sec'+str(j+1)+'_Lon_Ave'].iloc[spd_bin[j].astype(int)],\
                      coef[event_type+'_Sec'+str(j+1)+'_Lon_Std'].iloc[spd_bin[j].astype(int)])
            lonF_min_z[i,j] = z_Score(lonForce[(sec_idx[j]*stepSize).astype(int):(sec_idx[j+1]*stepSize).astype(int)].min(),\
                      coef[event_type+'_Sec'+str(j+1)+'_Lon_Ave'].iloc[spd_bin[j].astype(int)],\
                      coef[event_type+'_Sec'+str(j+1)+'_Lon_Std'].iloc[spd_bin[j].astype(int)])
            latF_max_z[i,j] = z_Score(latForce[(sec_idx[j]*stepSize).astype(int):(sec_idx[j+1]*stepSize).astype(int)].max(),\
                      coef[event_type+'_Sec'+str(j+1)+'_Lat_Ave'].iloc[spd_bin[j].astype(int)],\
                      coef[event_type+'_Sec'+str(j+1)+'_Lat_Std'].iloc[spd_bin[j].astype(int)])
            latF_min_z[i,j] = z_Score(latForce[(sec_idx[j]*stepSize).astype(int):(sec_idx[j+1]*stepSize).astype(int)].min(),\
                      coef[event_type+'_Sec'+str(j+1)+'_Lat_Ave'].iloc[spd_bin[j].astype(int)],\
                      coef[event_type+'_Sec'+str(j+1)+'_Lat_Std'].iloc[spd_bin[j].astype(int)])
                
    #Update event DataFrame  
    sec_1_spd = pd.DataFrame(spd_ave[:,0], columns=['Sec1_Speed(km/hr)'])
    sec_1_lon_max_z = pd.DataFrame(lonF_max_z[:,0], columns=['Sec1_Lon_Acc_Z'])
    sec_1_lon_min_z = pd.DataFrame(lonF_min_z[:,0], columns=['Sec1_Lon_Dec_Z'])
    sec_1_lat_max_z = pd.DataFrame(latF_max_z[:,0], columns=['Sec1_Lat_LT_Z'])
    sec_1_lat_min_z = pd.DataFrame(latF_min_z[:,0], columns=['Sec1_Lat_RT_Z'])
    sec_2_spd = pd.DataFrame(spd_ave[:,1], columns=['Sec2_Speed(km/hr)'])
    sec_2_lon_max_z = pd.DataFrame(lonF_max_z[:,1], columns=['Sec2_Lon_Acc_Z'])
    sec_2_lon_min_z = pd.DataFrame(lonF_min_z[:,1], columns=['Sec2_Lon_Dec_Z'])
    sec_2_lat_max_z = pd.DataFrame(latF_max_z[:,1], columns=['Sec2_Lat_LT_Z'])
    sec_2_lat_min_z = pd.DataFrame(latF_min_z[:,1], columns=['Sec2_Lat_RT_Z'])
    sec_3_spd = pd.DataFrame(spd_ave[:,2], columns=['Sec3_Speed(km/hr)'])
    sec_3_lon_max_z = pd.DataFrame(lonF_max_z[:,2], columns=['Sec3_Lon_Acc_Z'])
    sec_3_lon_min_z = pd.DataFrame(lonF_min_z[:,2], columns=['Sec3_Lon_Dec_Z'])
    sec_3_lat_max_z = pd.DataFrame(latF_max_z[:,2], columns=['Sec3_Lat_LT_Z'])
    sec_3_lat_min_z = pd.DataFrame(latF_min_z[:,2], columns=['Sec3_Lat_RT_Z'])

    features=['Sec1_Speed(km/hr)','Sec1_Lon_Acc_Z','Sec1_Lon_Dec_Z','Sec1_Lat_LT_Z','Sec1_Lat_RT_Z',\
              'Sec2_Speed(km/hr)','Sec2_Lon_Acc_Z','Sec2_Lon_Dec_Z','Sec2_Lat_LT_Z','Sec2_Lat_RT_Z',\
              'Sec3_Speed(km/hr)','Sec3_Lon_Acc_Z','Sec3_Lon_Dec_Z','Sec3_Lat_LT_Z','Sec3_Lat_RT_Z']   
    df_section = pd.concat([sec_1_spd, sec_1_lon_max_z, sec_1_lon_min_z, sec_1_lat_max_z, sec_1_lat_min_z,\
                            sec_2_spd, sec_2_lon_max_z, sec_2_lon_min_z, sec_2_lat_max_z, sec_2_lat_min_z,\
                            sec_3_spd, sec_3_lon_max_z, sec_3_lon_min_z, sec_3_lat_max_z, sec_3_lat_min_z], axis=1)
    df_section.columns = features            
    
    df_eva = pd.merge(df_event, df_section, how='left', left_index=True, right_index=True)        
        
    stop = timeit.default_timer()
    print ("Event Evaluation Run Time: %s seconds " % round((stop - start),2))
    
    return df_eva







