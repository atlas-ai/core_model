#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:02:27 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import distributions


def event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate):
    """ detect movement events based on rotation rate of z-axis and changes in course

    :param rot_rate_z: imu rotation rate around z   
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param course: gps course in radians
    :param spd: speed in m/s2
    :param evt_param: coefficients of detection model
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :return: preliminary detected events
    """
    dataLen = rot_z.shape[0]
    dataPoints = 20 #The number of data points to detect an event
    scanStep = int(samp_rate//10) #1/10 of a second    
    rotz_threshold = 0.05
    
    #Create empty data frame to store event data (RTT, LTT, LCR, LCL)
    df_event = pd.DataFrame(np.nan, index=np.arange(10000), columns=['type','prob','d',\
                            's_utc','e_utc','event_acc','s_spd','e_spd','s_crs','e_crs',\
                            's_lat','e_lat','s_long','e_long','s_alt','e_alt'])  
    
    event_no = 1
           
    #TOP LOOP:
    #Loop through event type (RTT, LTT, LCR, LCL)
    event_max_prob = np.zeros(4)               
    for k in range(4): 
        
        #Initialise scan parameters and probability threshold
        if k==0 or k==1:
            beg_window = int(5*(samp_rate//dataPoints))
            end_window = int(15*(samp_rate//dataPoints))
            num_of_window = 11
            pro_threshold = 0.8            
        elif k==2 or k==3:
            beg_window = int(2*(samp_rate//dataPoints))
            end_window = int(8*(samp_rate//dataPoints))
            num_of_window = 7
            pro_threshold = 0.6
        
        #MIDDLE Loop:
        #Loop through different scanning window sizes to capture the space for event patterns
        #Step size is used to define the length of scanning windows (2 indicates 1 seconds; 4 indicates 2 seconds; ... 30 indicates 15 seconds) 
        for stepSize in np.linspace(beg_window, end_window, num=num_of_window):
        
            windowSize = dataPoints*stepSize.astype(int)
            previous_event = windowSize
            event_max_prob = 0.0
            beg_idx = 0
            end_idx = 0
            beg_utc = 0.0
            end_utc = 0.0
            beg_lat = 0.0
            end_lat = 0.0
            beg_long = 0.0
            end_long = 0.0
            beg_alt = 0.0
            end_alt = 0.0
            beg_spd = 0.0
            end_spd = 0.0
            beg_crs = 0.0
            end_crs = 0.0
        
            #BOTTOM LOOP:
            #Loop through scan steps, which is 0.1 seconds.
            #Calculate the probabilities of four events and select the one with the highest probability.  
            for i in range(0, dataLen-windowSize, scanStep):
            
                #Create an empty array to hold independent variables
                dataVar = np.zeros(dataPoints*3+1)
                dataVar[0] = 1.0
            
                #Extract values of key data points for a window segment
                for j in range(1, dataPoints+1): 
                
                    idx = (i+(j-1)*stepSize).astype(int)                
                    dataVar[j] = rot_z.iloc[idx]
                    dataVar[j+dataPoints] = crs.iloc[idx]-crs.iloc[idx-1]
                    dataVar[j+2*dataPoints] = rot_z.index.values[idx]
                
                #Rotation w.r.t. z-axis must be close to zero to indicate the beginning and end of a event
                rotz_beg = dataVar[1]
                rotz_end = dataVar[20] 
            
                #Calculate probability for data segment
                event_prob = distributions.predict_prob_sigmoid(dataVar[0:2 * dataPoints + 1], evt_param[k])
                                                              
                #Identify events with pre-defined criteria
                if ((event_prob >= pro_threshold) and (np.abs(rotz_beg) <= rotz_threshold) and (np.abs(rotz_end) <= rotz_threshold)):
                    #Check whether the detected event overlaps with previous event with the same duration
                    if i<=previous_event:
                        #Loop to record maximum probability
                        if event_prob >= event_max_prob:
                            event_max_prob = event_prob
                            beg_utc = dataVar[2*dataPoints+1]
                            beg_idx = i
                            beg_lat = lat.iloc[beg_idx]
                            beg_long = long.iloc[beg_idx]
                            beg_alt = alt.iloc[beg_idx]
                            beg_spd = spd.iloc[beg_idx]
                            beg_crs = crs.iloc[beg_idx]
                            end_utc = dataVar[3*dataPoints]
                            end_idx = i+windowSize
                            end_lat = lat.iloc[end_idx]
                            end_long = long.iloc[end_idx]
                            end_alt = alt.iloc[end_idx]
                            end_spd = spd.iloc[end_idx]
                            end_crs = crs.iloc[end_idx]
                    else:
                        #Data entry when time of event changes
                        if k==0:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'RTT'
                        elif k==1:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'LTT'
                        elif k==2:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'LCR'
                        elif k==3:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'LCL'               
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_utc')] = beg_utc
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_utc')] = end_utc
                        df_event.iloc[event_no-1, df_event.columns.get_loc('d')] = (end_utc-beg_utc)                       
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_lat')] = beg_lat
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_lat')] = end_lat
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_long')] = beg_long
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_long')] = end_long
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_alt')] = beg_alt
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_alt')] = end_alt
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_spd')] = beg_spd
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_spd')] = end_spd
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_crs')] = beg_crs
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_crs')] = end_crs
                        df_event.iloc[event_no-1, df_event.columns.get_loc('prob')] = event_max_prob  
                        event_no += 1 
                    
                        #Set values for a new event
                        event_max_prob = event_prob
                        beg_utc = dataVar[2*dataPoints+1]
                        beg_idx = i
                        beg_lat = lat.iloc[beg_idx]
                        beg_long = long.iloc[beg_idx]
                        beg_alt = alt.iloc[beg_idx]
                        beg_spd = spd.iloc[beg_idx]
                        beg_crs = crs.iloc[beg_idx]
                        end_utc = dataVar[3*dataPoints]
                        end_idx = i+windowSize
                        end_lat = lat.iloc[end_idx]
                        end_long = long.iloc[end_idx]
                        end_alt = alt.iloc[end_idx]
                        end_spd = spd.iloc[end_idx]
                        end_crs = crs.iloc[end_idx]
                        previous_event = i+windowSize
             
            #Data entry when step size changes
            if event_max_prob!=0.0:    
                if k==0:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'RTT'
                elif k==1:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'LTT'
                elif k==2:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'LCR'
                elif k==3:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('type')] = 'LCL'
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_utc')] = beg_utc
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_utc')] = end_utc
                df_event.iloc[event_no-1, df_event.columns.get_loc('d')] = (end_utc-beg_utc)
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_lat')] = beg_lat
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_lat')] = end_lat
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_long')] = beg_long
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_long')] = end_long
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_alt')] = beg_alt
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_alt')] = end_alt
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_spd')] = beg_spd
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_spd')] = end_spd
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_crs')] = beg_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_crs')] = end_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('prob')] = event_max_prob    
                event_no += 1 
                                 
    df_event = df_event[df_event['prob'] > pro_threshold]               
    df_event = df_event.sort_values(['type', 'e_utc', 'prob'], ascending=[True, True, False])  
    df_event = df_event.reset_index(drop=True)  
    df_event['d'] = df_event['d']/1000000000
    df_event['s_utc'] = pd.to_datetime(df_event['s_utc']/1000000000, unit='s')
    df_event['e_utc'] = pd.to_datetime(df_event['e_utc']/1000000000, unit='s')
    df_event['event_acc'] = (df_event['e_spd']-df_event['s_spd'])/df_event['d']/3.6/9.8
       
    return df_event


def event_summary(df_event):
    """ remove duplicates and summarise event detection results

    :param df_event: preliminary dataframe for events   
    :return: detected events
    """
    
    if df_event.empty==False:
        
        #Selecting process to remove overlaps with same event types
        eventLen = df_event.shape[0]
        beg_utc = df_event['s_utc'].iloc[0]
        end_utc = df_event['e_utc'].iloc[0]
        type_idx = df_event['type'].iloc[0]
        df_event['overlap'] = 0
        overlap_idx = 1        
        for i in range(eventLen):
            if df_event['type'].iloc[i]!=type_idx: 
                beg_utc = df_event['s_utc'].iloc[i]
                end_utc = df_event['e_utc'].iloc[i]
                type_idx = df_event['type'].iloc[i]
                overlap_idx += 1
            if df_event['s_utc'].iloc[i] > end_utc: 
                end_utc = df_event['e_utc'].iloc[i]
                overlap_idx += 1
                df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
            else:
                if (((end_utc-df_event['s_utc'].iloc[i])/(end_utc-beg_utc)>=1/3) or\
                    ((end_utc-df_event['s_utc'].iloc[i])/(df_event['e_utc'].iloc[i]-df_event['s_utc'].iloc[i])>=1/3)):
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
                else:
                    end_utc = df_event['e_utc'].iloc[i]
                    overlap_idx += 1
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx                                                  
                
        df_event = df_event.loc[df_event.reset_index().groupby(['overlap'])['prob'].idxmax()]
        df_event = df_event.reset_index(drop=True) 
        df_event = df_event.drop('overlap', axis=1)
        
        #Repeat selecting process to remove overlaps with different event types
        df_event = df_event.sort_values(['e_utc', 'prob'], ascending=[True, False])  
        df_event = df_event.reset_index(drop=True)
        eventLen = df_event.shape[0]
        beg_utc = df_event['s_utc'].iloc[0]
        end_utc = df_event['e_utc'].iloc[0]
        type_idx = df_event['type'].iloc[0]
        df_event['overlap'] = 0
        overlap_idx = 1
        for i in range(eventLen):
            if df_event['s_utc'].iloc[i] > end_utc: 
                end_utc = df_event['e_utc'].iloc[i]
                overlap_idx += 1
                df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
            else:
                if (((end_utc-df_event['s_utc'].iloc[i])/(end_utc-beg_utc)>=1/3) or\
                    ((end_utc-df_event['s_utc'].iloc[i])/(df_event['e_utc'].iloc[i]-df_event['s_utc'].iloc[i])>=1/3)):
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
                else:
                    end_utc = df_event['e_utc'].iloc[i]
                    overlap_idx += 1
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx  
              
        df_event = df_event.loc[df_event.reset_index().groupby(['overlap'])['prob'].idxmax()]
        df_event = df_event.reset_index(drop=True) 
        df_event = df_event.drop('overlap', axis=1)
            
    return df_event                


def excess_acc_detection(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, z_threshold):
    """ detect the excess acceleration or deceleration 
    
    :param acc_x: longitudinal force of a vehicle
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param crs: course in radians
    :param spd: speed of a vehicle in km/hr
    :param acc_param: coefficients of detection model
    :samp_rate: sampling rate of raw data (has to be the multiple of 20)
    :param z_threshold: threshold of z-score that acceleration breaches
    :return: data frame to summarise the occasions of excess acceleration
    """    
    df_acc_sum = pd.DataFrame(np.nan, index=np.arange(10000), columns=['type','prob','d',\
                            's_utc','e_utc','event_acc','s_spd','e_spd','s_crs','e_crs',\
                            's_lat','e_lat','s_long','e_long','s_alt','e_alt',\
                            'duplicate'])

    acc_num = 0
    max_acc = 0
    max_dec = 0
              
    dataLen = len(acc_x)
    scanStep = int(samp_rate//10) #1/10 of a second  
    
    for i in range(1, dataLen, scanStep):
        
        if acc_x[i] > (acc_param['acc_ave'][0]+z_threshold*np.sqrt(acc_param['acc_var'][0])):
            if (acc_x[i]>=max_acc) and (acc_x[i]>=acc_x[i-1]):
                max_acc=acc_x[i]
                max_utc=acc_x.index.values[i] 
            elif (acc_x[i]<max_acc) and (acc_x[i]<acc_x[i-1]):
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('type')] = 'EXA'
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_utc')] = max_utc
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_spd')] = spd[i-1]
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('event_acc')] = max_acc
                acc_num += 1
                max_acc = 0

        elif acc_x[i] < (acc_param['dec_ave'][0]-z_threshold*np.sqrt(acc_param['dec_var'][0])):
            if (acc_x[i]<=max_dec) and (acc_x[i]<=acc_x[i-1]):
                max_dec=acc_x[i]
                max_utc=acc_x.index.values[i] 
            elif (acc_x[i]>max_dec) and (acc_x[i]>acc_x[i-1]):
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('type')] = 'EXD'
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_utc')] = max_utc
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_spd')] = spd[i-1]
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('event_acc')] = max_dec
                acc_num += 1
                max_dec = 0
    
    df_acc_sum = df_acc_sum.dropna(how='all')
    
    #remove duplicate records
    if df_acc_sum.empty==False:
        
        df_acc_sum['duplicate']=0
        accLen=df_acc_sum.shape[0]
        df_acc_sum.iloc[0,df_acc_sum.columns.get_loc('duplicate')]=1
        overlap_indicator = 1
        df_acc_sum['event_acc']=df_acc_sum['event_acc'].apply(lambda x: -1*x if x<0 else x)
                
        for i in range(1, accLen):
            
            if df_acc_sum['type'][i]==df_acc_sum['type'][i-1]:
                if (df_acc_sum['e_utc'][i]-df_acc_sum['e_utc'][i-1])/np.timedelta64(1, 's')<=1:
                    df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('duplicate')]=overlap_indicator
                else:
                    overlap_indicator += 1
                    df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('duplicate')]=overlap_indicator
            else:
                overlap_indicator += 1
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('duplicate')]=overlap_indicator
        
        df_acc_sum = df_acc_sum.loc[df_acc_sum.groupby('duplicate')['event_acc'].idxmax()]
        
        df_acc_sum = df_acc_sum.reset_index(drop=True)
        accLen = df_acc_sum.shape[0]
        for i in range(accLen):
            if df_acc_sum['type'][i]=='EXD':
                temp_max_dec = df_acc_sum['event_acc'][i]
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('event_acc')]=-1*temp_max_dec
        
        df_acc_sum = df_acc_sum.drop('duplicate',axis=1)
                       
        #expand maximum acc from a point to a period
        t_shift = int(2.5*samp_rate)
        accLen = df_acc_sum.shape[0]
        for i in range(accLen):
            idx = acc_x.index.searchsorted(df_acc_sum['e_utc'][i]) 
            if (idx>(t_shift-1)) and ((idx+t_shift)<len(acc_x)):
                s_utc = acc_x.index.values[idx-t_shift]
                s_lat = lat[idx-t_shift]
                s_long = long[idx-t_shift]
                s_alt = alt[idx-t_shift]
                s_spd = spd[idx-t_shift]
                s_crs = crs[idx-t_shift]
                e_utc = acc_x.index.values[idx+t_shift]
                e_lat = lat[idx+t_shift]
                e_long = long[idx+t_shift]
                e_alt = alt[idx+t_shift]
                e_spd = spd[idx+t_shift]
                e_crs = crs[idx+t_shift]
                for j in range (idx, (idx-t_shift), -1):
                    if (acc_x[j]<0.01) and (acc_x[j]>-0.01):
                        s_utc = acc_x.index.values[j]
                        s_lat = lat[j]
                        s_long = long[j]
                        s_alt = alt[j]
                        s_spd = spd[j]
                        s_crs = crs[j]
                        break
                for j in range (idx, (idx+t_shift), 1):
                    if (acc_x[j]<0.01) and (acc_x[j]>-0.01):
                        e_utc = acc_x.index.values[j]
                        e_lat = lat[j]
                        e_long = long[j]
                        e_alt = alt[j]
                        e_spd = spd[j]
                        e_crs = crs[j]
                        break

                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_utc')]=s_utc
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_lat')]=s_lat
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_long')]=s_long
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_alt')]=s_alt               
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_spd')]=s_spd
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_crs')]=s_crs
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_utc')]=e_utc
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_lat')]=e_lat
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_long')]=e_long
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_alt')]=e_alt
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_spd')]=e_spd
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_crs')]=e_crs        
                duration = (e_utc-s_utc)/np.timedelta64(1, 's')
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('d')]=duration

        df_acc_sum['prob']=1.0
        df_acc_sum = df_acc_sum[df_acc_sum['d']>0]
        
    return df_acc_sum





