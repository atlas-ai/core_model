#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:02:27 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import distributions


def event_detection(rot_z, crs, spd, param):
    
    dataLen = rot_z.shape[0]
    dataPoints = 20
    scanStep = 2    
    rotz_threshold = 0.03
    
    #Create empty data frame to store event data (RTT, LTT, LCR, LCL)
    df_event = pd.DataFrame(np.nan, index=np.arange(1000), columns=['type','d','s_idx','e_idx','s_timestamp','e_timestamp',\
                            's_spd','e_spd','ave_acc','s_crs','e_crs','prob'])  
    
    event_no = 1
           
    #TOP LOOP:
    #Loop through event type (RTT, LTT, LCR, LCL)
    event_max_prob = np.zeros(4)               
    for k in range(4): 
        
        #Initialise scan parameters and probability threshold
        if k==0 or k==1:
            beg_window = 5
            end_window = 15
            num_of_window = 11
            pro_threshold = 0.9            
        elif k==2 or k==3:
            beg_window = 2
            end_window = 8
            num_of_window = 7
            pro_threshold = 0.7
        
        #MIDDLE Loop:
        #Loop through different scanning window sizes to capture the space for event patterns
        #Step size is used to define the length of scanning windows (2 indicates 1 seconds; 4 indicates 2 seconds; ... 30 indicates 15 seconds) 
        for stepSize in np.linspace(beg_window, end_window, num=num_of_window):
        
            windowSize = dataPoints*stepSize.astype(int)
            previous_event = windowSize
            event_max_prob = 0.0
            beg_idx=0
            end_idx=0
            beg_timestamp = 0.0
            end_timestamp = 0.0            
            beg_spd=0.0
            end_spd=0.0
            beg_crs=0.0
            end_crs=0.0
        
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
                event_prob = distributions.predict_prob_sigmoid(dataVar[0:2 * dataPoints + 1], param[k])
                                            
                #Identify events with pre-defined criteria
                if ((event_prob >= pro_threshold) and (np.abs(rotz_beg) <= rotz_threshold) and (np.abs(rotz_end) <= rotz_threshold)):
                    #Check whether the detected event overlaps with previous event with the same duration
                    if i<=previous_event:
                        #Loop to record maximum probability
                        if event_prob >= event_max_prob:
                            event_max_prob = event_prob
                            beg_timestamp = dataVar[2*dataPoints+1]
                            beg_idx = i
                            beg_spd = spd.iloc[i]
                            beg_crs = crs.iloc[i]
                            end_timestamp = dataVar[3*dataPoints]
                            end_idx = i+windowSize
                            end_spd = spd.iloc[i+windowSize]
                            end_crs = crs.iloc[i+windowSize]
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
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_idx')] = beg_idx
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_idx')] = end_idx
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_timestamp')] = beg_timestamp
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_timestamp')] = end_timestamp
                        df_event.iloc[event_no-1, df_event.columns.get_loc('d')] = stepSize
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_spd')] = beg_spd
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_spd')] = end_spd
                        df_event.iloc[event_no-1, df_event.columns.get_loc('ave_acc')] = (end_spd-beg_spd)/(stepSize/2)/3.6/9.8
                        df_event.iloc[event_no-1, df_event.columns.get_loc('s_crs')] = beg_crs
                        df_event.iloc[event_no-1, df_event.columns.get_loc('e_crs')] = end_crs
                        df_event.iloc[event_no-1, df_event.columns.get_loc('prob')] = event_max_prob                         
                        event_no += 1 
                    
                        #Set values for a new event
                        event_max_prob = event_prob
                        beg_timestamp = dataVar[2*dataPoints+1]
                        beg_idx = i
                        beg_spd = spd.iloc[i]
                        beg_crs = crs.iloc[i]
                        end_timestamp = dataVar[3*dataPoints]
                        end_idx = i+windowSize
                        end_spd = spd.iloc[i+windowSize]
                        end_crs = crs.iloc[i+windowSize]
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
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_idx')] = beg_idx
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_idx')] = end_idx
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_timestamp')] = beg_timestamp
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_timestamp')] = end_timestamp
                df_event.iloc[event_no-1, df_event.columns.get_loc('d')] = stepSize
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_spd')] = beg_spd
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_spd')] = end_spd
                df_event.iloc[event_no-1, df_event.columns.get_loc('ave_acc')] = (end_spd-beg_spd)/(stepSize/2)/3.6/9.8
                df_event.iloc[event_no-1, df_event.columns.get_loc('s_crs')] = beg_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('e_crs')] = end_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('prob')] = event_max_prob  
                event_no += 1 
                                 
    df_event = df_event[df_event['prob'] > pro_threshold]               
    df_event = df_event.sort_values(['type', 'e_idx', 'prob'], ascending=[True, True, False])  
    df_event = df_event.reset_index(drop=True)   
    print ("\n")
    print ("Detection of events without removing overlaps:")
    print (df_event)
       
    return df_event


def event_summary(df_event):
    
    if df_event.empty==False:
        
        #Selecting process to remove overlaps with same event types
        eventLen = df_event.shape[0]
        startRow = df_event['s_idx'].iloc[0]
        endRow = df_event['e_idx'].iloc[0]
        type_idx = df_event['type'].iloc[0]
        df_event['overlap'] = 0
        overlap_idx = 1        
        for i in range(eventLen):
            if df_event['type'].iloc[i]!=type_idx: 
                startRow = df_event['s_idx'].iloc[i]
                endRow = df_event['e_idx'].iloc[i]
                type_idx = df_event['type'].iloc[i]
                overlap_idx += 1
            if df_event['s_idx'].iloc[i] > endRow: 
                endRow = df_event['e_idx'].iloc[i]
                overlap_idx += 1
                df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
            else:
                if (((endRow-df_event['s_idx'].iloc[i])/(endRow-startRow)>=1/3) or\
                    ((endRow-df_event['s_idx'].iloc[i])/(df_event['e_idx'].iloc[i]-df_event['s_idx'].iloc[i])>=1/3)):
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
                else:
                    endRow = df_event['e_idx'].iloc[i]
                    overlap_idx += 1
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx                                                  
                
        df_event = df_event.loc[df_event.reset_index().groupby(['overlap'])['prob'].idxmax()]
        df_event = df_event.reset_index(drop=True) 
        df_event = df_event.drop('overlap', axis=1)
        print ("\n")
        print ("Detection of events: Remove overlaps with same event types ")
        print (df_event)
        
        #Repeat selecting process to remove overlaps with different event types
        df_event = df_event.sort_values(['e_timestamp', 'prob'], ascending=[True, False])  
        df_event = df_event.reset_index(drop=True)
        eventLen = df_event.shape[0]
        startRow = df_event['s_idx'].iloc[0]
        endRow = df_event['e_idx'].iloc[0]
        type_idx = df_event['type'].iloc[0]
        df_event['overlap'] = 0
        overlap_idx = 1
        for i in range(eventLen):
            if df_event['s_idx'].iloc[i] > endRow: 
                endRow = df_event['e_idx'].iloc[i]
                overlap_idx += 1
                df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
            else:
                if (((endRow-df_event['s_idx'].iloc[i])/(endRow-startRow)>=1/3) or\
                    ((endRow-df_event['s_idx'].iloc[i])/(df_event['e_idx'].iloc[i]-df_event['s_idx'].iloc[i])>=1/3)):
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx
                else:
                    endRow = df_event['e_idx'].iloc[i]
                    overlap_idx += 1
                    df_event.iloc[i, df_event.columns.get_loc('overlap')] = overlap_idx  
              
        df_event = df_event.loc[df_event.reset_index().groupby(['overlap'])['prob'].idxmax()]
        df_event = df_event.reset_index(drop=True) 
        df_event = df_event.drop('overlap', axis=1)
        df_event['s_idx'] = df_event['s_idx'].astype(int)
        df_event['e_idx'] = df_event['e_idx'].astype(int)
        print ("\n")
        print ("Detection of events: Remove overlaps with different event types ")
        print (df_event)
        
        df_event['s_utc'] = pd.to_datetime(df_event['s_timestamp']/1000000000, unit='s')
        df_event['e_utc'] = pd.to_datetime(df_event['e_timestamp']/1000000000, unit='s')
        df_event = df_event.drop(['s_idx','e_idx','s_timestamp','e_timestamp'], axis=1)
        df_event = df_event[['type','d','s_utc','e_utc','s_spd','e_spd','ave_acc','s_crs','e_crs','prob']]
        print ("\n")
        print ("Detection of events: Final table ")
        print (df_event)
            
    return df_event                


def excess_acc_detection(acc_x, crs, spd, df_param, z_threshold):
    """ detect the excess acceleration or deceleration 
    
    :param acc_x: longitudinal force of a vehicle
    :param spd: speed of a vehicle in km/hr
    :param z_threshold: threshold of z-score that acceleration breaches
    :return: data frame to summarise the occasions of excess acceleration
    """    
    df_acc_sum = pd.DataFrame(np.nan, index=np.arange(10000), columns=['type','d','s_utc','e_utc',\
                              's_spd','e_spd','ave_acc','s_crs','e_crs','max_acc','score','duplicate'])

    acc_num = 0
    max_acc = 0
    max_dec = 0
        
    dataLen = len(acc_x)
    for i in range(1,dataLen):
        
        if acc_x[i] > (df_param['acc_ave'][0]+z_threshold*np.sqrt(df_param['acc_var'][0])):
            if (acc_x[i]>=max_acc) and (acc_x[i]>=acc_x[i-1]):
                max_acc=acc_x[i]
                max_utc=acc_x.index.values[i] 
            elif (acc_x[i]<max_acc) and (acc_x[i]<acc_x[i-1]):
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('type')] = 'EXA'
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_utc')] = max_utc
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_spd')] = spd[i-1]
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('max_acc')] = max_acc
                acc_num += 1
                max_acc = 0

        elif acc_x[i] < (df_param['dec_ave'][0]-z_threshold*np.sqrt(df_param['dec_var'][0])):
            if (acc_x[i]<=max_dec) and (acc_x[i]<=acc_x[i-1]):
                max_dec=acc_x[i]
                max_utc=acc_x.index.values[i] 
            elif (acc_x[i]>max_dec) and (acc_x[i]>acc_x[i-1]):
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('type')] = 'EXD'
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_utc')] = max_utc
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('e_spd')] = spd[i-1]
                df_acc_sum.iloc[acc_num, df_acc_sum.columns.get_loc('max_acc')] = max_dec
                acc_num += 1
                max_dec = 0
    
    df_acc_sum = df_acc_sum.dropna(how='all')
    print ("\n")
    print ("Detection of acceleration: Initial screening ")
    print (df_acc_sum)
    
    #remove duplicate records
    if df_acc_sum.empty==False:
        
        df_acc_sum['duplicate']=0
        accLen=df_acc_sum.shape[0]
        df_acc_sum.iloc[0,df_acc_sum.columns.get_loc('duplicate')]=1
        overlap_indicator = 1
        df_acc_sum['max_acc']=df_acc_sum['max_acc'].apply(lambda x: -1*x if x<0 else x)
                
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
        
        df_acc_sum = df_acc_sum.loc[df_acc_sum.groupby('duplicate')['max_acc'].idxmax()]
        
        df_acc_sum = df_acc_sum.reset_index(drop=True)
        accLen = df_acc_sum.shape[0]
        for i in range(accLen):
            if df_acc_sum['type'][i]=='EXD':
                temp_max_dec = df_acc_sum['max_acc'][i]
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('max_acc')]=-1*temp_max_dec
        
        df_acc_sum = df_acc_sum.drop('duplicate',axis=1)
        print ("\n")
        print ("Detection of acceleration: Point ")
        print (df_acc_sum)
        
        #expand maximum acc from a point to a period
        accLen = df_acc_sum.shape[0]
        for i in range(accLen):
            idx = acc_x.index.searchsorted(df_acc_sum['e_utc'][i])            
            if (idx>49) and ((idx+50)<len(acc_x)):
                s_utc = acc_x.index.values[idx-50]
                s_spd = spd[idx-50]
                s_crs = crs[idx-50]
                e_utc = acc_x.index.values[idx+50]
                e_spd = spd[idx+50]
                e_crs = crs[idx+50]
                for j in range (idx, (idx-50), -1):
                    if (acc_x[j]<0.01) and (acc_x[j]>-0.01):
                        s_utc = acc_x.index.values[j]
                        s_spd = spd[j]
                        s_crs = crs[j]
                        break
                for j in range (idx, (idx+50), 1):
                    if (acc_x[j]<0.01) and (acc_x[j]>-0.01):
                        e_utc = acc_x.index.values[j]
                        e_spd = spd[j]
                        e_crs = crs[j]
                        break

                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_utc')]=s_utc
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_spd')]=s_spd
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('s_crs')]=s_crs
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_utc')]=e_utc
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_spd')]=e_spd
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('e_crs')]=e_crs        
                duration = (e_utc-s_utc)/np.timedelta64(1, 's')
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('d')]=duration
                df_acc_sum.iloc[i,df_acc_sum.columns.get_loc('ave_acc')]= (e_spd-s_spd)/duration/3.6/9.8
        
        print ("\n")
        print ("Detection of acceleration: Period ")
        print (df_acc_sum)
        
    return df_acc_sum





