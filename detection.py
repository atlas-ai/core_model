 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:02:27 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import distributions


def original_event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, turn_threshold, lane_change_threshold):
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
    rotz_threshold = 0.03
    
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
            end_window = int(18*(samp_rate//dataPoints))
            num_of_window = 14
            pro_threshold = turn_threshold            
        elif k==2 or k==3:
            beg_window = int(2*(samp_rate//dataPoints))
            end_window = int(8*(samp_rate//dataPoints))
            num_of_window = 7
            pro_threshold = lane_change_threshold
        
        #MIDDLE Loop:
        #Loop through different scanning window sizes to capture the space for event patterns
        #Step size is used to define the length of scanning windows 
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
                dataVar = np.empty(dataPoints*3+1)
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
                event_prob = distributions.predict_prob_sigmoid(dataVar[0:2*dataPoints+1], evt_param[k])
                                                              
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
                        #print('time of event change')
                        #print('beg_idx: %s; end_idx: %s; d_idx: %s; beg_utc: %s; end_utc: %s; d: %s' % (beg_idx, end_idx,(end_idx-beg_idx)/20,\
                        #                                                              pd.to_datetime(beg_utc/1000000000, unit='s'),\
                        #                                                              pd.to_datetime(end_utc/1000000000, unit='s'),\
                        #                                                              (end_utc-beg_utc)/1000000000))
                        #print(dataVar[1:dataPoints+1])
                        #print(dataVar[dataPoints+1:2*dataPoints+1])
                        #print(pd.to_datetime(dataVar[2*dataPoints+1:3*dataPoints+1]/1000000000,unit='s'))
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
                #print('step size change')
                #print('beg_idx: %s; end_idx: %s; d_idx: %s; beg_utc: %s; end_utc: %s; d: %s' % (beg_idx, end_idx,(end_idx-beg_idx)/20,\
                #                                                                      pd.to_datetime(beg_utc/1000000000, unit='s'),\
                #                                                                      pd.to_datetime(end_utc/1000000000, unit='s'),\
                #                                                                      (end_utc-beg_utc)/1000000000))
                #print(dataVar[1:dataPoints+1])
                #print(dataVar[dataPoints+1:2*dataPoints+1])
                #print(pd.to_datetime(dataVar[2*dataPoints+1:3*dataPoints+1]/1000000000,unit='s'))
                event_no += 1 
                                 
    df_event = df_event[df_event['prob'] > pro_threshold]               
    df_event = df_event.sort_values(['type', 'e_utc', 'prob'], ascending=[True, True, False])  
    df_event = df_event.reset_index(drop=True)  
    df_event['d'] = df_event['d']/1000000000
    df_event['s_utc'] = pd.to_datetime(df_event['s_utc']/1000000000, unit='s')
    df_event['e_utc'] = pd.to_datetime(df_event['e_utc']/1000000000, unit='s')
    df_event['event_acc'] = (df_event['e_spd']-df_event['s_spd'])/df_event['d']/3.6/9.8
    #print(df_event)
       
    return df_event


def remove_evt_duplicates(df_event):
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
        #print('Summary of evernts')
        #print(df_event)
        
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
    z_threshold = z_threshold + 2 #Adjustments due to change of sampling rate
    df_acc_sum = pd.DataFrame(np.nan, index=np.arange(10000), columns=['type','prob','d',\
                            's_utc','e_utc','event_acc','s_spd','e_spd','s_crs','e_crs',\
                            's_lat','e_lat','s_long','e_long','s_alt','e_alt',\
                            'duplicate'])

    acc_num = 0
    max_acc = 0.0
    max_dec = 0.0
              
    dataLen = len(acc_x)
    scanStep = int(samp_rate//10) #1/10 of a second  
    
    for i in range(1, dataLen, scanStep):
        
        if acc_x[i] > (acc_param['acc_ave'][0]+z_threshold*np.sqrt(acc_param['acc_var'][0])):
            if (acc_x[i]>=max_acc) and (acc_x[i]>=acc_x[i-1]):
                max_acc=acc_x[i]
                max_utc=acc_x.index[i] 
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
                max_utc=acc_x.index[i] 
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
                s_utc = acc_x.index[idx-t_shift]
                s_lat = lat[idx-t_shift]
                s_long = long[idx-t_shift]
                s_alt = alt[idx-t_shift]
                s_spd = spd[idx-t_shift]
                s_crs = crs[idx-t_shift]
                e_utc = acc_x.index[idx+t_shift]
                e_lat = lat[idx+t_shift]
                e_long = long[idx+t_shift]
                e_alt = alt[idx+t_shift]
                e_spd = spd[idx+t_shift]
                e_crs = crs[idx+t_shift]
                for j in range (idx, (idx-t_shift), -1):
                    if (acc_x[j]<0.01) and (acc_x[j]>-0.01):
                        s_utc = acc_x.index[j]
                        s_lat = lat[j]
                        s_long = long[j]
                        s_alt = alt[j]
                        s_spd = spd[j]
                        s_crs = crs[j]
                        break
                for j in range (idx, (idx+t_shift), 1):
                    if (acc_x[j]<0.01) and (acc_x[j]>-0.01):
                        e_utc = acc_x.index[j]
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


# Pre-screening functions

def thresholding_algo(y, lag, threshold, influence, avg, std):
    # Use fixed thresholds as boundaries to generate signals (initialise avg and std)
    df = pd.DataFrame(np.nan, index=y.index, columns=['Signal', 'Filter_Mean', 'Filter_Std'])
    if len(y)>lag:
        signals = np.zeros(len(y))
        filteredY = np.asarray(y)
        avgFilter = np.zeros(len(y))
        stdFilter = np.zeros(len(y))
        avgFilter[lag - 1] = avg
        stdFilter[lag - 1] = std
        for i in range(lag, len(y)):
            if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
                if y[i] > avgFilter[i - 1]:
                    signals[i] = 1
                else:
                    signals[i] = -1

                filteredY[i] = (1 - influence) * filteredY[i - 1] + influence * y[i]
                avgFilter[i] = (1 - influence) * avg + influence * np.mean(filteredY[(i - lag):i])
                stdFilter[i] = (1 - influence) * std + influence * np.std(filteredY[(i - lag):i])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = (1 - influence) * avg + influence * np.mean(filteredY[(i - lag):i])
                stdFilter[i] = (1 - influence) * std + influence * np.std(filteredY[(i - lag):i])

        df['Signal'] = signals
        df['Filter_Mean'] = avgFilter
        df['Filter_Std'] = stdFilter

    return df


def transformation_algo(signals, samp_rate):
    
    trans = pd.DataFrame(np.nan, index=np.arange(1000), columns=['s_time', 'e_time', 'dur', 'gap', 'turn_code'])

    trans_sig = signals['Signal'] + 2 * signals['Signal'].diff()

    idx = 0
    pattern = np.zeros((2, 2))
    sig_len = len(trans_sig)
    for i in range(1, sig_len):
        if (trans_sig[i] == 3.) or (trans_sig[i] == -3.):  # define starting time
            pattern[0, 0] = i
            pattern[1, 0] = trans_sig[i]
        elif (trans_sig[i] == 2.) or (trans_sig[i] == -2.):  # define ending time
            pattern[0, 1] = i
            pattern[1, 1] = trans_sig[i]
            # write dataframe of 'trans'
            # create turn_code: right turn = 1; left turn = -1; lane change to right = 5; lane change to left = -5.
            trans.iloc[idx, trans.columns.get_loc('turn_code')] = pattern[1].sum()
            if (pattern[0, 0] - 3 * samp_rate) < 0:
                trans.iloc[idx, trans.columns.get_loc('s_time')] = trans_sig.index[0]
            else:
                trans.iloc[idx, trans.columns.get_loc('s_time')] = trans_sig.index[int(pattern[0, 0] - 3 * samp_rate)]
            if (pattern[0, 1] + 3 * samp_rate) > (sig_len - 1):
                trans.iloc[idx, trans.columns.get_loc('e_time')] = trans_sig.index[int(sig_len - 1)]
            else:
                trans.iloc[idx, trans.columns.get_loc('e_time')] = trans_sig.index[int(pattern[0, 1] + 3 * samp_rate)]
            # reset pattern
            pattern = np.zeros((2, 2))
            idx += 1
    trans = trans.dropna(how='all')
    if trans.empty==False:
        trans['dur'] = (trans['e_time'] - trans['s_time']) / np.timedelta64(1, 's')
        if trans.shape[0]>1:
            trans['gap'] = (trans['s_time'] - trans['e_time'].shift(1)) / np.timedelta64(1, 's')
        trans.iloc[0, trans.columns.get_loc('gap')] = 0
        
    return trans


def segmentation_algo(trans):
    
    seg = pd.DataFrame(np.nan, index=np.arange(1000), columns=['s_time', 'e_time', 'dur', 'seg_code'])
    # seg_code: right turn = 1; left turn = 2; lane change (right) = 3; lane change (left) = 4; others = 5
    if trans.empty==False:
        idx = 0
        seg.iloc[idx, seg.columns.get_loc('s_time')] = trans['s_time'][0]
        seg.iloc[idx, seg.columns.get_loc('e_time')] = trans['e_time'][0]
        if trans['turn_code'][0] == 1.:
            seg.iloc[idx, seg.columns.get_loc('seg_code')] = 1.
        elif trans['turn_code'][0] == -1.:
            seg.iloc[idx, seg.columns.get_loc('seg_code')] = 2.
        elif trans['turn_code'][0] == 5.:
            seg.iloc[idx, seg.columns.get_loc('seg_code')] = 3.
        elif trans['turn_code'][0] == -5.:
            seg.iloc[idx, seg.columns.get_loc('seg_code')] = 4.
        else:
            seg.iloc[idx, seg.columns.get_loc('seg_code')] = 5.

        trans_len = trans.shape[0]
        for i in range(1, trans_len):
            if trans['gap'][i] <= 0:
                seg.iloc[idx, seg.columns.get_loc('e_time')] = trans['e_time'][i]
                if (seg['seg_code'][idx] == 1.) and (trans['turn_code'][i] == 1.):
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 1.
                elif (seg['seg_code'][idx] == -1.) and (trans['turn_code'][i] == -1.):
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 2.
                elif (seg['seg_code'][idx] == 1.) and (trans['turn_code'][i] == -1.):
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 3.
                elif (seg['seg_code'][idx] == -1.) and (trans['turn_code'][i] == 1.):
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 4.
                else:
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 5.
            else:
                idx += 1
                seg.iloc[idx, seg.columns.get_loc('s_time')] = trans['s_time'][i]
                seg.iloc[idx, seg.columns.get_loc('e_time')] = trans['e_time'][i]
                if trans['turn_code'][i] == 1.:
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 1.
                elif trans['turn_code'][i] == -1.:
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 2.
                elif trans['turn_code'][i] == 5.:
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 3.
                elif trans['turn_code'][i] == -5.:
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 4.
                else:
                    seg.iloc[idx, seg.columns.get_loc('seg_code')] = 5.
        
        seg['dur'] = (seg['e_time'] - seg['s_time']) / np.timedelta64(1, 's')
        
    seg = seg.dropna(how='all')
    return seg


def evt_det_algo(rot_z, lat, long, alt, crs, spd, param, samp_rate, threshold, seg_code):
    
    dataLen = rot_z.shape[0]
    dataPoints = 20  # The number of data points to detect an event
    scanStep = int(samp_rate // 10)  # 1/10 of a second
    rotz_threshold = 0.03

    # Create empty data frame to store event data (RTT, LTT, LCR, LCL)
    df_event = pd.DataFrame(np.nan, index=np.arange(100), columns=['type', 'prob', 'd', \
                                                                   's_utc', 'e_utc', 'event_acc', 's_spd', 'e_spd',\
                                                                   's_crs', 'e_crs', \
                                                                   's_lat', 'e_lat', 's_long', 'e_long', 's_alt',\
                                                                   'e_alt'])
    event_no = 1
    pro_threshold = threshold

    # Initialise scan parameters and probability threshold
    if seg_code == 1 or seg_code == 2:
        beg_window = int(5 * (samp_rate // dataPoints))
        end_window = int(18 * (samp_rate // dataPoints))
        num_of_window = 14
    elif seg_code == 3 or seg_code == 4:
        beg_window = int(2 * (samp_rate // dataPoints))
        end_window = int(8 * (samp_rate // dataPoints))
        num_of_window = 7

    for stepSize in np.linspace(beg_window, end_window, num=num_of_window):

        windowSize = dataPoints * stepSize.astype(int)
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

        for i in range(0, dataLen - windowSize, scanStep):

            # Create an empty array to hold independent variables
            dataVar = np.empty(dataPoints * 3 + 1)
            dataVar[0] = 1.0

            # Extract values of key data points for a window segment
            for j in range(1, dataPoints + 1):
                idx = (i + (j - 1) * stepSize).astype(int)
                dataVar[j] = rot_z.iloc[idx]
                dataVar[j + dataPoints] = crs.iloc[idx] - crs.iloc[idx - 1]
                dataVar[j + 2 * dataPoints] = rot_z.index.values[idx]

            # Rotation w.r.t. z-axis must be close to zero to indicate the beginning and end of a event
            rotz_beg = dataVar[1]
            rotz_end = dataVar[20]

            # Calculate probability for data segment
            event_prob = distributions.predict_prob_sigmoid(dataVar[0:2 * dataPoints + 1], param)

            # Identify events with pre-defined criteria
            if ((event_prob >= pro_threshold) and (np.abs(rotz_beg) <= rotz_threshold) and \
                        (np.abs(rotz_end) <= rotz_threshold)):
                # Check whether the detected event overlaps with previous event with the same duration
                if i <= previous_event:
                    # Loop to record maximum probability
                    if event_prob >= event_max_prob:
                        event_max_prob = event_prob
                        beg_utc = dataVar[2 * dataPoints + 1]
                        beg_idx = i
                        beg_lat = lat.iloc[beg_idx]
                        beg_long = long.iloc[beg_idx]
                        beg_alt = alt.iloc[beg_idx]
                        beg_spd = spd.iloc[beg_idx]
                        beg_crs = crs.iloc[beg_idx]
                        end_utc = dataVar[3 * dataPoints]
                        end_idx = i + windowSize
                        end_lat = lat.iloc[end_idx]
                        end_long = long.iloc[end_idx]
                        end_alt = alt.iloc[end_idx]
                        end_spd = spd.iloc[end_idx]
                        end_crs = crs.iloc[end_idx]
                else:
                    # Data entry when time of event changes
                    if seg_code == 1:
                        df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'RTT'
                    elif seg_code == 2:
                        df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'LTT'
                    elif seg_code == 3:
                        df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'LCR'
                    elif seg_code == 4:
                        df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'LCL'
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('s_utc')] = beg_utc
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('e_utc')] = end_utc
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('d')] = (end_utc - beg_utc)
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('s_lat')] = beg_lat
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('e_lat')] = end_lat
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('s_long')] = beg_long
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('e_long')] = end_long
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('s_alt')] = beg_alt
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('e_alt')] = end_alt
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('s_spd')] = beg_spd
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('e_spd')] = end_spd
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('s_crs')] = beg_crs
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('e_crs')] = end_crs
                    df_event.iloc[event_no - 1, df_event.columns.get_loc('prob')] = event_max_prob
                    event_no += 1

                    # Set values for a new event
                    event_max_prob = event_prob
                    beg_utc = dataVar[2 * dataPoints + 1]
                    beg_idx = i
                    beg_lat = lat.iloc[beg_idx]
                    beg_long = long.iloc[beg_idx]
                    beg_alt = alt.iloc[beg_idx]
                    beg_spd = spd.iloc[beg_idx]
                    beg_crs = crs.iloc[beg_idx]
                    end_utc = dataVar[3 * dataPoints]
                    end_idx = i + windowSize
                    end_lat = lat.iloc[end_idx]
                    end_long = long.iloc[end_idx]
                    end_alt = alt.iloc[end_idx]
                    end_spd = spd.iloc[end_idx]
                    end_crs = crs.iloc[end_idx]
                    previous_event = i + windowSize

        # Data entry when step size changes
        if event_max_prob != 0.0:
            if seg_code == 1:
                df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'RTT'
            elif seg_code == 2:
                df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'LTT'
            elif seg_code == 3:
                df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'LCR'
            elif seg_code == 4:
                df_event.iloc[event_no - 1, df_event.columns.get_loc('type')] = 'LCL'
            df_event.iloc[event_no - 1, df_event.columns.get_loc('s_utc')] = beg_utc
            df_event.iloc[event_no - 1, df_event.columns.get_loc('e_utc')] = end_utc
            df_event.iloc[event_no - 1, df_event.columns.get_loc('d')] = (end_utc - beg_utc)
            df_event.iloc[event_no - 1, df_event.columns.get_loc('s_lat')] = beg_lat
            df_event.iloc[event_no - 1, df_event.columns.get_loc('e_lat')] = end_lat
            df_event.iloc[event_no - 1, df_event.columns.get_loc('s_long')] = beg_long
            df_event.iloc[event_no - 1, df_event.columns.get_loc('e_long')] = end_long
            df_event.iloc[event_no - 1, df_event.columns.get_loc('s_alt')] = beg_alt
            df_event.iloc[event_no - 1, df_event.columns.get_loc('e_alt')] = end_alt
            df_event.iloc[event_no - 1, df_event.columns.get_loc('s_spd')] = beg_spd
            df_event.iloc[event_no - 1, df_event.columns.get_loc('e_spd')] = end_spd
            df_event.iloc[event_no - 1, df_event.columns.get_loc('s_crs')] = beg_crs
            df_event.iloc[event_no - 1, df_event.columns.get_loc('e_crs')] = end_crs
            df_event.iloc[event_no - 1, df_event.columns.get_loc('prob')] = event_max_prob
            event_no += 1

    df_event = df_event[df_event['prob'] > pro_threshold]
    df_event = df_event.sort_values(['type', 'e_utc', 'prob'], ascending=[True, True, False])
    df_event = df_event.reset_index(drop=True)
    df_event['d'] = df_event['d'] / 1000000000
    df_event['s_utc'] = pd.to_datetime(df_event['s_utc'] / 1000000000, unit='s')
    df_event['e_utc'] = pd.to_datetime(df_event['e_utc'] / 1000000000, unit='s')
    df_event['event_acc'] = (df_event['e_spd'] - df_event['s_spd']) / df_event['d'] / 3.6 / 9.8

    return df_event


def event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, turn_threshold, lane_change_threshold):
    df_evt = pd.DataFrame(np.nan, index=np.arange(0), columns=['type', 'prob', 'd', \
                                                               's_utc', 'e_utc', 'event_acc', 's_spd', 'e_spd',
                                                               's_crs', 'e_crs', \
                                                               's_lat', 'e_lat', 's_long', 'e_long', 's_alt', 'e_alt'])
    # Step 1: Pre-screen to obtain signal segments
    rot_z_avg = rot_z.where(np.abs(rot_z) <= 0.02).mean()
    rot_z_std = rot_z.where(np.abs(rot_z) <= 0.02).std()
    signals = thresholding_algo(rot_z, lag=5, threshold=6, influence=0.01, avg=rot_z_avg, std=rot_z_std)

    trans = transformation_algo(signals, samp_rate=50)
    seg = segmentation_algo(trans)
    
    # Step 2: Run detection model
    if seg.empty==False:
        seg_no = seg.shape[0]
        for i in range(seg_no):
            st = seg['s_time'][i]
            et = seg['e_time'][i]
            sc = seg['seg_code'][i]

            rot_z_seg = rot_z[st:et]
            lat_seg = lat[st:et]
            long_seg = long[st:et]
            alt_seg = alt[st:et]
            crs_seg = crs[st:et]
            spd_seg = spd[st:et]

            if sc == 1.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, threshold=turn_threshold, seg_code=sc)
                df_evt = df_evt.append(df)
            elif sc == 2.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, threshold=turn_threshold, seg_code=sc)
                df_evt = df_evt.append(df)
            elif sc == 3.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, threshold=lane_change_threshold, seg_code=sc)
                df_evt = df_evt.append(df)
            elif sc == 4.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, threshold=lane_change_threshold, seg_code=sc)
                df_evt = df_evt.append(df)
            elif sc == 5.:
                for j in range(4):
                    if j == 0 or j == 1:
                        evt_threshold = turn_threshold
                    elif j == 2 or j == 3:
                        evt_threshold = lane_change_threshold
                    df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                                 evt_param[j], samp_rate, threshold=evt_threshold, seg_code=(j + 1))
                    df_evt = df_evt.append(df)

        df_evt = df_evt.reset_index(drop=True)

    return df_evt
