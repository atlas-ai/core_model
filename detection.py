#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:02:27 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import quaternion_extra as fmt
import distributions


def read_param(csv_file):
    param = np.asarray(pd.read_csv(csv_file, index_col=0)).transpose()
    return param


def read_df(df):
    #df = pd.read_excel(file_name)
    #df.set_index(['t'], inplace=True)
    df = df.dropna(how='all')
    rot_z = df['r_rate_z']
    crs = df['course']
    spd = df['speed']
    return rot_z, crs, spd


def Event_Detection(rot_z, crs, spd, param):
    
    dataLen = rot_z.shape[0]
    dataPoints = 50
    scanStep = 10    
    rotz_threshold = 0.02
    
    #Create empty data frame to store event data (RTT, LTT, LCR, LCL)
    df_event = pd.DataFrame(np.nan, index=np.arange(1000), columns=['Type','Duration(s)','Start_Index','End_Index','Start_Timestamp','End_Timestamp',\
                            'Start_Spd(km/h)','End_Spd(km/h)','Acceleration(m/s2)','Start_Course','End_Course','Probability'])    
    event_no = 1
           
    #TOP LOOP:
    #Loop through event type (RTT, LTT, LCR, LCL)
    event_max_prob = np.zeros(4)               
    for k in range(4): 
        
        #Initialise scan parameters and probability threshold
        if k==0 or k==1:
            beg_window = 10
            end_window = 30
            num_of_window = 11
            pro_threshold = 0.9            
        elif k==2 or k==3:
            beg_window = 4
            end_window = 16
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
                    dataVar[j+dataPoints] = crs.iloc[idx]
                    dataVar[j+2*dataPoints] = rot_z.index.values[idx]
            
                #Rotation w.r.t. z-axis must be close to zero to indicate the beginning and end of a event
                rotz_beg = dataVar[1]
                rotz_end = dataVar[50] 
            
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
                            df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'RTT'
                        elif k==1:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'LTT'
                        elif k==2:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'LCR'
                        elif k==3:
                            df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'LCL'
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Duration(s)')] = stepSize/2
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Index')] = beg_idx
                        df_event.iloc[event_no-1, df_event.columns.get_loc('End_Index')] = end_idx
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Timestamp')] = beg_timestamp
                        df_event.iloc[event_no-1, df_event.columns.get_loc('End_Timestamp')] = end_timestamp
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Spd(km/h)')] = beg_spd*3.6
                        df_event.iloc[event_no-1, df_event.columns.get_loc('End_Spd(km/h)')] = end_spd*3.6
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Acceleration(m/s2)')] = (end_spd-beg_spd)/(stepSize/2)
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Course')] = beg_crs
                        df_event.iloc[event_no-1, df_event.columns.get_loc('End_Course')] = end_crs
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Probability')] = event_max_prob                         
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
                    df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'RTT'
                elif k==1:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'LTT'
                elif k==2:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'LCR'
                elif k==3:
                    df_event.iloc[event_no-1, df_event.columns.get_loc('Type')] = 'LCL'
                df_event.iloc[event_no-1, df_event.columns.get_loc('Duration(s)')] = stepSize/2
                df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Index')] = beg_idx
                df_event.iloc[event_no-1, df_event.columns.get_loc('End_Index')] = end_idx
                df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Timestamp')] = beg_timestamp
                df_event.iloc[event_no-1, df_event.columns.get_loc('End_Timestamp')] = end_timestamp
                df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Spd(km/h)')] = beg_spd*3.6
                df_event.iloc[event_no-1, df_event.columns.get_loc('End_Spd(km/h)')] = end_spd*3.6
                df_event.iloc[event_no-1, df_event.columns.get_loc('Acceleration(m/s2)')] = (end_spd-beg_spd)/(stepSize/2)
                df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Course')] = beg_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('End_Course')] = end_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('Probability')] = event_max_prob  
                event_no += 1 
                                 
    df_event = df_event[df_event['Probability'] > pro_threshold]               
    df_event = df_event.sort_values(['Type', 'End_Timestamp', 'Probability'], ascending=[True, True, False])  
    df_event = df_event.reset_index(drop=True)   
       
    return df_event


def Event_Summary(df_event):
    
    if df_event.empty==False:
        
        #Selecting process to remove overlaps with same event types
        eventLen = df_event.shape[0]
        startRow = df_event['Start_Index'].iloc[0]
        endRow = df_event['End_Index'].iloc[0]
        type_idx = df_event['Type'].iloc[0]
        df_event['Overlap_Index'] = 0
        overlap_idx = 1        
        for i in range(eventLen):
            if df_event['Type'].iloc[i]!=type_idx: 
                startRow = df_event['Start_Index'].iloc[i]
                endRow = df_event['End_Index'].iloc[i]
                type_idx = df_event['Type'].iloc[i]
                overlap_idx += 1
            if df_event['Start_Index'].iloc[i] > endRow: 
                endRow = df_event['End_Index'].iloc[i]
                overlap_idx += 1
                df_event.iloc[i, df_event.columns.get_loc('Overlap_Index')] = overlap_idx
            else:
                if (((endRow-df_event['Start_Index'].iloc[i])/(endRow-startRow)>=1/3) or\
                    ((endRow-df_event['Start_Index'].iloc[i])/(df_event['End_Index'].iloc[i]-df_event['Start_Index'].iloc[i])>=1/3)):
                    df_event.iloc[i, df_event.columns.get_loc('Overlap_Index')] = overlap_idx
                else:
                    endRow = df_event['End_Index'].iloc[i]
                    overlap_idx += 1
                    df_event.iloc[i, df_event.columns.get_loc('Overlap_Index')] = overlap_idx                                                  
                
        df_event = df_event.loc[df_event.reset_index().groupby(['Overlap_Index'])['Probability'].idxmax()]
        df_event = df_event.reset_index(drop=True) 
        df_event = df_event.drop('Overlap_Index', axis=1)
        

        #Repeat selecting process to remove overlaps with different event types
        df_event = df_event.sort_values(['End_Timestamp', 'Probability'], ascending=[True, False])  
        df_event = df_event.reset_index(drop=True)
        eventLen = df_event.shape[0]
        startRow = df_event['Start_Index'].iloc[0]
        endRow = df_event['End_Index'].iloc[0]
        type_idx = df_event['Type'].iloc[0]
        df_event['Overlap_Index'] = 0
        overlap_idx = 1
        for i in range(eventLen):
            if df_event['Start_Index'].iloc[i] > endRow: 
                endRow = df_event['End_Index'].iloc[i]
                overlap_idx += 1
                df_event.iloc[i, df_event.columns.get_loc('Overlap_Index')] = overlap_idx
            else:
                if (((endRow-df_event['Start_Index'].iloc[i])/(endRow-startRow)>=1/3) or\
                    ((endRow-df_event['Start_Index'].iloc[i])/(df_event['End_Index'].iloc[i]-df_event['Start_Index'].iloc[i])>=1/3)):
                    df_event.iloc[i, df_event.columns.get_loc('Overlap_Index')] = overlap_idx
                else:
                    endRow = df_event['End_Index'].iloc[i]
                    overlap_idx += 1
                    df_event.iloc[i, df_event.columns.get_loc('Overlap_Index')] = overlap_idx  
              
        df_event = df_event.loc[df_event.reset_index().groupby(['Overlap_Index'])['Probability'].idxmax()]
        df_event = df_event.reset_index(drop=True) 
        df_event = df_event.drop('Overlap_Index', axis=1)
        df_event['Start_Index'] = df_event['Start_Index'].astype(int)
        df_event['End_Index'] = df_event['End_Index'].astype(int)
        
        df_event.to_csv('test.csv')
            
    return df_event                






