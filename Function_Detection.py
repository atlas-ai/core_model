# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 18:38:25 2017

@author: seanx
"""

import pandas as pd
import numpy as np
import timeit


######################################################
######  1. Function to detect driving events    ######
###################################################### 

def Event_Detection(df):
    start = timeit.default_timer()
    
##############################################################################################################################
#                                                          Initialisation                                                    #
##############################################################################################################################    
    
    param = np.asarray(pd.read_csv('Detection_Coefficients.csv', index_col=0)).transpose()
  
    att = df['attitude_sum']
    rot_z = df['rotation_rate_z(radians/s)']
    spd = df['speed(m/s)']*3.6
    crs =  df['course(degree)']
    
    dataLen = df.shape[0]
    dataPoints = 50
    scanStep = 10    
    rotz_threshold = 0.05

    #Define sigmoid function
    def sigmoid(t):                          
        return (1/(1 + np.e**(-t)))   
    
    #Define prediction function
    def predict_prob(X, theta):
        p = sigmoid(np.dot(X, theta))    
        return p
    
##############################################################################################################################
#                                                          Detection Loop                                                    #
##############################################################################################################################    

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
            pro_threshold = 0.8
        
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
            for i in range(0, dataLen-windowSize+1, scanStep):
            
                #Create an empty array to hold independent variables
                dataVar = np.zeros(dataPoints*3+1)
                dataVar[0] = 1.0
            
                #Extract values of key data points for a window segment
                for j in range(1, dataPoints+1): 
                
                    idx = (i+(j-1)*stepSize).astype(int)                
                    dataVar[j] = att.iloc[idx]
                    dataVar[j+dataPoints] = rot_z.iloc[idx]
                    dataVar[j+2*dataPoints] = att.index.values[idx]
            
                #Rotation w.r.t. z-axis must be close to zero to indicate the beginning and end of a event
                rotz_beg = dataVar[51]
                rotz_end = dataVar[100] 
            
                #Calculate probability for data segment
                event_prob = predict_prob(dataVar[0:2*dataPoints+1], param[k])
                                            
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
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Spd(km/h)')] = beg_spd
                        df_event.iloc[event_no-1, df_event.columns.get_loc('End_Spd(km/h)')] = end_spd
                        df_event.iloc[event_no-1, df_event.columns.get_loc('Acceleration(m/s2)')] = (end_spd-beg_spd)/3.6/(stepSize/2)
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
                df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Spd(km/h)')] = beg_spd
                df_event.iloc[event_no-1, df_event.columns.get_loc('End_Spd(km/h)')] = end_spd
                df_event.iloc[event_no-1, df_event.columns.get_loc('Acceleration(m/s2)')] = (end_spd-beg_spd)/3.6/(stepSize/2)
                df_event.iloc[event_no-1, df_event.columns.get_loc('Start_Course')] = beg_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('End_Course')] = end_crs
                df_event.iloc[event_no-1, df_event.columns.get_loc('Probability')] = event_max_prob  
                event_no += 1 
                                 
    df_event = df_event[df_event['Probability'] > pro_threshold]               
    df_event = df_event.sort_values(['Type', 'End_Timestamp', 'Probability'], ascending=[True, True, False])  
    df_event = df_event.reset_index(drop=True)   
    
    #print("Print data structure after all three loops:")
    #print(df_event)
    #print("\n")
    
##############################################################################################################################
#                                           Selection of Events with Maximum Probability                                     #
##############################################################################################################################    
    
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
        
        #print("Overlaps with same event types are removed:")
        #print(df_event)
        #print("\n")
        
        
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
        
        #print("Overlaps with different event types but same timestamps are removed:")
        #print(df_event)
        #print("\n")
        
    #df_event.to_csv('event.csv', index=False)
    
    stop = timeit.default_timer()
    print ("Event Detection Run Time: %s seconds " % round((stop - start),2))
                
    return df_event                


######################################################
######  2. Function to detect driving status    ######
###################################################### 
    
def Status_Detection(df, df_eva, sec_len):   
    start = timeit.default_timer()
    
##############################################################################################################################
#                                                          Initialisation                                                    #
##############################################################################################################################      
    
    spd = df['speed(m/s)']*3.6 #Convert m/s to km/h
    crs = df['course(degree)']
   
    dataLen = df.shape[0]
    scanStep = 50  
    spd_buffer = 2
    section_len = sec_len #minumum length of status
    
    #Create empty data frame to store status data (PRK, CRS)
    df_status = pd.DataFrame(np.nan, index=np.arange(500), columns=['Type','Duration(s)','Start_Index','End_Index',\
                             'Start_Timestamp','End_Timestamp','Start_Spd(km/h)','End_Spd(km/h)','Acceleration(m/s2)',\
                             'Start_Course','End_Course'])
    
    #Initialise arrays
    spd_ave = np.zeros(dataLen//scanStep+1)
    crs_ave = np.zeros(dataLen//scanStep+1)
    spd_change = np.zeros(dataLen//scanStep+1)
    turn_mark = np.zeros(dataLen//scanStep+1)
    timestamp = np.zeros(dataLen//scanStep+1)
    start_index = np.zeros(dataLen//scanStep+1)
    
    #Loop through average speed to find inflection points
    idx=0
    for i in range(0, dataLen+1, scanStep):
        spd_ave[idx] = spd.iloc[i:(i+scanStep+1)].mean()
        crs_ave[idx] = crs.iloc[i:(i+scanStep+1)].mean()
        start_index[idx] = i
        timestamp[idx] = spd.index.values[i]
        if spd_ave[idx]==0:
            spd_change[idx] = 99 #stop
        else:
            if idx==0:
                spd_change[idx] = 0
            else:
                if spd_ave[idx]>spd_ave[idx-1]+spd_buffer:
                    spd_change[idx] = 1 #speed up
                elif spd_ave[idx]<spd_ave[idx-1]-spd_buffer:
                    spd_change[idx] = 2 #slow down
                else:
                    spd_change[idx] = 0
        idx += 1
       
    #df_spd = pd.DataFrame({'ave_spd':spd_ave, 'spd_change':spd_change})
    #print("Status Detection: Find inflection points based on changes of speed")
    #print(df_spd)
    
    #Loop through inflection points to get the upward or downward trends
    spdLen = len(spd_ave)
    spd_index = np.zeros(spdLen)
    beg_spd = spd_ave[0]
    beg_idx = 0
    for i in range(spdLen):
        if spd_change[i]!=0:
            if spd_ave[i]==0:
                spd_index[i] = 99
                beg_idx = i+1
                beg_spd = spd_ave[i]
            else:
                if spd_ave[i]>beg_spd+spd_buffer:
                    for j in range(beg_idx,i+1):
                        spd_index[j] = 1
                    beg_idx = i+1
                    beg_spd = spd_ave[i]
                elif spd_ave[i]<beg_spd-spd_buffer:
                    for j in range(beg_idx,i+1):
                        spd_index[j] = 2
                    beg_idx = i+1
                    beg_spd = spd_ave[i]
                else:
                    for j in range(beg_idx,i+1):
                        spd_index[j] = 0
                    beg_idx = i+1
                    beg_spd = spd_ave[i]
                
    #df_spd = pd.DataFrame({'ave_spd':spd_ave, 'spd_change':spd_change, 'trend_index':spd_index})
    #print("Status Detection: Get trend of speed")
    #print(df_spd)
    
    #Loop through trend to remove small fluctuation from trends
    section_count = 0
    beg_section = spd_index[0]
    spd_count = np.zeros(spdLen)
    previous_spd_index = spd_index[0]
    for i in range(spdLen):
        if spd_index[i] == beg_section:
            section_count += 1
        else:
            spd_count[i-section_count:i]=section_count
            section_count = 1 
            beg_section = spd_index[i]
    for i in range(spdLen):
        if spd_count[i]>section_len:
            previous_spd_index = spd_index[i]
        else:
            spd_index[i]=previous_spd_index
    
    #Exclude short stops
    #for i in range(spdLen):
    #    if spd_ave[i]==0:
    #        spd_index[i]=99
            
    #df_spd = pd.DataFrame({'ave_spd':spd_ave, 'spd_change':spd_change, 'trend_index':spd_index})
    #print("Status Detection: Remove small changes in speed")
    #print(df_spd)
    
    #Select 'right turn' or 'left turn' data only
    if df_eva.empty==False:    
        df_rlt = df_eva.loc[(df_eva['Type']=='RTT') | (df_eva['Type']=='LTT')]
        #Loop through to remove Right Turns and Left Turns (marked 5)   
        eventLen = df_rlt.shape[0]
        if eventLen!=0:
            evt_no = 0
            evt_beg = df_rlt['Start_Timestamp'].iloc[0]
            evt_end = df_rlt['End_Timestamp'].iloc[0]
            for i in range(spdLen):
                if evt_no<eventLen:
                    if timestamp[i]<evt_beg:
                        turn_mark[i] = 0
                    else:
                        turn_mark[i] = 1
                        if timestamp[i]>evt_end:
                            turn_mark[i]=0
                            if evt_no+1<eventLen:
                                evt_no += 1
                                evt_beg = df_rlt['Start_Timestamp'].iloc[evt_no]
                                evt_end = df_rlt['End_Timestamp'].iloc[evt_no]
            for i in range(spdLen):
                if turn_mark[i]==1:
                    spd_index[i]=5
    
    #df_spd = pd.DataFrame({'ave_spd':spd_ave, 'spd_change':spd_change, 'trend_index':spd_index, 'turn_mark':turn_mark})
    #print("Status Detection: Remove turn data from status")
    #print(df_spd)

##############################################################################################################################
#                                                    Create Output DataFrame                                                 #
##############################################################################################################################  
   
    #Loop through to obtain summary DataFrame
    section_count = 0
    beg_section = spd_index[0]
    beg_spd = spd_ave[0]
    beg_crs = crs_ave[0]
    beg_timestamp = timestamp[0]
    beg_index = start_index[0]
    df_idx = 0
    for i in range(spdLen):
        if spd_index[i] == beg_section:
            section_count += 1
        else:
            end_crs = crs_ave[i-1]
            if spd_index[i-1]!=99:
                #Correct course direction                
                if end_crs-beg_crs>270:
                    beg_crs = beg_crs+360
                elif end_crs-beg_crs<-270:
                    end_crs = end_crs+360
                #Identify course direction    
                if (end_crs-beg_crs>-30) and (end_crs-beg_crs<30):
                    df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'CRS'
                elif end_crs-beg_crs<=-30:
                    df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'CRL'
                elif end_crs-beg_crs>=30:
                    df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'CRR'  
                #Identify turns
                if spd_index[i-1]==5:
                    df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'TRN'
            else:
                df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'PRK'                    
            df_status.iloc[df_idx, df_status.columns.get_loc('Duration(s)')] = (section_count*scanStep/100)
            df_status.iloc[df_idx, df_status.columns.get_loc('Start_Index')] = beg_index
            df_status.iloc[df_idx, df_status.columns.get_loc('End_Index')] = start_index[i]
            df_status.iloc[df_idx, df_status.columns.get_loc('Start_Timestamp')] = beg_timestamp
            df_status.iloc[df_idx, df_status.columns.get_loc('End_Timestamp')] = timestamp[i]
            df_status.iloc[df_idx, df_status.columns.get_loc('Start_Spd(km/h)')] = beg_spd
            df_status.iloc[df_idx, df_status.columns.get_loc('End_Spd(km/h)')] = spd_ave[i-1]
            df_status.iloc[df_idx, df_status.columns.get_loc('Acceleration(m/s2)')] = (spd_ave[i-1]-beg_spd)/3.6/(section_count*scanStep/100)      
            df_status.iloc[df_idx, df_status.columns.get_loc('Start_Course')] = beg_crs
            df_status.iloc[df_idx, df_status.columns.get_loc('End_Course')] = end_crs
            df_idx += 1
            section_count = 1
            beg_section = spd_index[i]
            beg_spd = spd_ave[i]
            beg_crs = crs_ave[i]
            beg_timestamp = timestamp[i]
            beg_index = start_index[i]    
    
    end_crs = crs_ave[spdLen-1]
    if spd_index[spdLen-1]!=99:
        #Correct course direction        
        if end_crs-beg_crs>270:
            beg_crs = beg_crs+360
        elif end_crs-beg_crs<-270:
            end_crs = end_crs+360
        #Identify course direction    
        if (end_crs-beg_crs>-30) and (end_crs-beg_crs<30):
            df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'CRS'
        elif end_crs-beg_crs<=-30:
            df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'CRL'
        elif end_crs-beg_crs>=30:
            df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'CRR'
        #Identify turns
        if spd_index[i-1]==5:
            df_status.iloc[df_idx, df_status.columns.get_loc('Type')] = 'TRN'
    else:
        df_status['Type'].iloc[df_idx] = 'PRK'
    df_status.iloc[df_idx, df_status.columns.get_loc('Duration(s)')]  = (section_count*scanStep/100)
    df_status.iloc[df_idx, df_status.columns.get_loc('Start_Index')] = beg_index
    df_status.iloc[df_idx, df_status.columns.get_loc('End_Index')] = start_index[spdLen-1]
    df_status.iloc[df_idx, df_status.columns.get_loc('Start_Timestamp')] = beg_timestamp
    df_status.iloc[df_idx, df_status.columns.get_loc('End_Timestamp')] = timestamp[spdLen-1]   
    df_status.iloc[df_idx, df_status.columns.get_loc('Start_Spd(km/h)')] = beg_spd
    df_status.iloc[df_idx, df_status.columns.get_loc('End_Spd(km/h)')] = spd_ave[spdLen-2]
    df_status.iloc[df_idx, df_status.columns.get_loc('Acceleration(m/s2)')] = (spd_ave[spdLen-2]-beg_spd)/3.6/(section_count*scanStep/100)
    df_status.iloc[df_idx, df_status.columns.get_loc('Start_Course')] = beg_crs
    df_status.iloc[df_idx, df_status.columns.get_loc('End_Course')] = end_crs
    
    df_status = df_status.dropna()
    df_status = df_status[df_status['Type']!='TRN']
    df_status = df_status.reset_index(drop=True) 
    
    #df_status.to_csv('status.csv', index=False)
    
    stop = timeit.default_timer()
    print ("Status Detection Run Time: %s seconds " % round((stop - start),2))
                
    return df_status    
    
    
######################################################
######  3. Function to detect emergency break   ######
######################################################    
    
def Braking_Detection(df):
    start = timeit.default_timer()
    
    
    df_break = pd.DataFrame(np.nan, index=np.arange(1), columns=['Type','Duration(s)','Start_Index','End_Index',\
                             'Start_Timestamp','End_Timestamp','Start_Spd(km/h)','End_Spd(km/h)','Acceleration(m/s2)',\
                             'Start_Course','End_Course']) 
    
    #Pending for further actions
    
    
    df_break = df_break.dropna()
        
    stop = timeit.default_timer()
    print ("Emergency Breaking Detection Run Time: %s seconds " % round((stop - start),2))
    
    return df_break    
    
    
    
    
    
    
    
    
    
    