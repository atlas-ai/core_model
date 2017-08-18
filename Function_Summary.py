# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:48:31 2017

@author: seanx
"""

import pandas as pd
import numpy as np
from datetime import datetime
import timeit

########################################################################################################
######  1. Function to combine event, status and breaking data to generate a summary data sheet   ######
########################################################################################################   

def Data_Summary(df_eva, df_status, df_break):
    start = timeit.default_timer()
    
    features = ['Type','Duration(s)','Start_Index','End_Index','Start_Timestamp','End_Timestamp',\
                'Start_Spd(km/h)','End_Spd(km/h)','Acceleration(m/s2)','Start_Course','End_Course','Probability',\
                'Sec1_Speed(km/hr)','Sec1_Lon_Acc_Z','Sec1_Lon_Dec_Z','Sec1_Lat_LT_Z','Sec1_Lat_RT_Z',\
                'Sec2_Speed(km/hr)','Sec2_Lon_Acc_Z','Sec2_Lon_Dec_Z','Sec2_Lat_LT_Z','Sec2_Lat_RT_Z',\
                'Sec3_Speed(km/hr)','Sec3_Lon_Acc_Z','Sec3_Lon_Dec_Z','Sec3_Lat_LT_Z','Sec3_Lat_RT_Z']
    
    post_features = ['Type','Duration(s)','Start_Timestamp','End_Timestamp','Start_Spd(km/h)','End_Spd(km/h)',\
                     'Acceleration(m/s2)','Start_Course','End_Course','Probability',\
                     'Sec1_Speed(km/hr)','Sec1_Lon_Acc_Z','Sec1_Lon_Dec_Z','Sec1_Lat_LT_Z','Sec1_Lat_RT_Z',\
                     'Sec2_Speed(km/hr)','Sec2_Lon_Acc_Z','Sec2_Lon_Dec_Z','Sec2_Lat_LT_Z','Sec2_Lat_RT_Z',\
                     'Sec3_Speed(km/hr)','Sec3_Lon_Acc_Z','Sec3_Lon_Dec_Z','Sec3_Lat_LT_Z','Sec3_Lat_RT_Z']
    
    df_sum = pd.DataFrame(np.NaN, index=np.arange(1), columns=features)
    df_sum = df_sum.dropna()
    
    df_sum = df_sum.append(df_status, ignore_index=True)    
    
    if df_break.empty==False:
        df_sum = df_sum.append(df_break, ignore_index=True)
    
    if df_eva.empty==False:
        df_sum = df_sum.append(df_eva, ignore_index=True)          
        
    df_sum = df_sum[features]    
    df_sum['Start_Index'] = df_sum['Start_Index'].astype(int)
    df_sum['End_Index'] = df_sum['End_Index'].astype(int)
    df_sum = df_sum.sort_values(['Start_Timestamp'], ascending=[True])
    df_sum = df_sum.reset_index(drop=True)     
    
    df_sum['Score'] = 0.0
    Tot_Score = 100.0    
    sumLen = df_sum.shape[0]
    for i in range (sumLen):
        if df_sum['Type'].iloc[i]=='RTT' or df_sum['Type'].iloc[i]=='LTT' or df_sum['Type'].iloc[i]=='LCR' or df_sum['Type'].iloc[i]=='LCL':                    
            event_score = Event_Score(df_sum.iloc[i])
            df_sum.iloc[i, df_sum.columns.get_loc('Score')] = event_score
            Tot_Score = Tot_Score - event_score
        
    df_sum = df_sum[post_features]
    
    stop = timeit.default_timer()
    print ("Output Summary Run Time: %s seconds \n" % round((stop - start),2))
    
    return df_sum, Tot_Score

########################################################################################################
######  2. Function to calculate scores of individual events without display                      ######
######################################################################################################## 

def Event_Score(event_record):
        
    score = 0.0
    if event_record['Type']=='RTT' or event_record['Type']=='LTT':
        for i in range(3):
            #Acceleration/Deceleration 
            if event_record['Sec'+ str(i+1) +'_Lon_Acc_Z'] > -1*event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']:
                if event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']>3 and event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']<=6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']/3*2
                elif event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']>6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']/3*3
            elif event_record['Sec'+ str(i+1) +'_Lon_Acc_Z'] < -1*event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']:
                if event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']<-3 and event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']>=-6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']/(-3)*2
                elif event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']<-6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']/(-3)*3
            #Lateral Force
            if event_record['Type']=='LTT':
                if event_record['Sec'+ str(i+1) +'_Lat_LT_Z']>3 and event_record['Sec'+ str(i+1) +'_Lat_LT_Z']<=6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_LT_Z']/3*2
                elif event_record['Sec'+ str(i+1) +'_Lat_LT_Z']>6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_LT_Z']/3*3
            elif event_record['Type']=='RTT':
                if event_record['Sec'+ str(i+1) +'_Lat_RT_Z']<-3 and event_record['Sec'+ str(i+1) +'_Lat_RT_Z']>=-6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_RT_Z']/(-3)*2
                elif event_record['Sec'+ str(i+1) +'_Lat_RT_Z']<-6:
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_RT_Z']/(-3)*3
    elif event_record['Type']=='LCR' or event_record['Type']=='LCL':    
        #Acceleration/Deceleration 
        LC_Lon = np.max([event_record['Sec1_Lon_Acc_Z'], -1*event_record['Sec1_Lon_Dec_Z'], event_record['Sec2_Lon_Acc_Z'], -1*event_record['Sec2_Lon_Dec_Z']])
        if event_record['Start_Spd(km/h)']<event_record['End_Spd(km/h)']:
            if LC_Lon>3 and LC_Lon<=6:
                score = score + LC_Lon/3*2
            elif LC_Lon>6:
                score = score + LC_Lon/3*3
        elif event_record['Start_Spd(km/h)']>event_record['End_Spd(km/h)']:
            if LC_Lon>3 and LC_Lon<=6:
                score = score + LC_Lon/3*2
            elif LC_Lon>6:
                score = score + LC_Lon/3*3
        #Lateral Force        
        LC_Lat = np.max([event_record['Sec1_Lat_LT_Z'], -1*event_record['Sec1_Lat_RT_Z'], event_record['Sec2_Lat_LT_Z'], -1*event_record['Sec2_Lat_RT_Z']])     
        if LC_Lat>3 and LC_Lat<=6:
            score = score + LC_Lat/3*2
        elif LC_Lat>6: 
            score = score + LC_Lat/3*3
    
    return round(score,0)


########################################################################################################
######  3. Function to make a query on events and display detailed analysis (Algorithm same as 2) ######
######################################################################################################## 

def Event_Query(event_record):
    
    Diag_spd = ['You were entering the turn at an average of ','You were making the turn at an average of ','You were exiting the turn at an average of ']
    Diag_Lon = ['Smooth','Slightly hard','Too hard and please be mindful']
    Diag_Lat_RT = ['Smooth','Slightly tilted to the right','Tilted to the right too much']
    Diag_Lat_LT = ['Smooth','Slightly tilted to the left','Tilted to the left too much']
    Diag_Lane = ['smoothly','fairly quickly','too abruptly and please be mindful']
    
    beg_time = datetime.fromtimestamp(event_record['Start_Timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    
    if event_record['Type']=='RTT':
        print('Right Turn Profile at %s' % beg_time)
    elif event_record['Type']=='LTT':
        print('Left Turn Profile at %s' % beg_time)
    elif event_record['Type']=='LCR':
        print('Lane Change to Right Profile at %s' % beg_time)
    elif event_record['Type']=='LCL':
        print('Lane Change to Left Profile at %s' % beg_time)
        
    score = 0.0
    if event_record['Type']=='RTT' or event_record['Type']=='LTT':
        for i in range(3):
            if event_record['Type']=='RTT':
                print(Diag_spd[i] + str(event_record['Sec'+ str(i+1) +'_Speed(km/hr)'].round(0)) + ' km/hr.')
            elif event_record['Type']=='LTT':
                print(Diag_spd[i] + str(event_record['Sec'+ str(i+1) +'_Speed(km/hr)'].round(0)) + ' km/hr.')
            #Acceleration/Deceleration 
            if event_record['Sec'+ str(i+1) +'_Lon_Acc_Z'] > -1*event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']:
                if event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']<=3:
                    print('Acceleration: ' + Diag_Lon[0])
                elif event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']>3 and event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']<=6:
                    print('Acceleration: ' + Diag_Lon[1])
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']/3*2
                elif event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']>6:
                    print('Acceleration: ' + Diag_Lon[2])
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Acc_Z']/3*3
            elif event_record['Sec'+ str(i+1) +'_Lon_Acc_Z'] < -1*event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']:
                if event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']>=-3:
                    print('Deceleration: ' + Diag_Lon[0])
                elif event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']<-3 and event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']>=-6:
                    print('Deceleration: ' + Diag_Lon[1])
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']/(-3)*2
                elif event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']<-6:
                    print('Deceleration: ' + Diag_Lon[2])
                    score = score + event_record['Sec'+ str(i+1) +'_Lon_Dec_Z']/(-3)*3
            #Lateral Force
            if event_record['Type']=='LTT':
                if event_record['Sec'+ str(i+1) +'_Lat_LT_Z']<=3:
                    print('Lateral Force: ' + Diag_Lat_LT[0])
                elif event_record['Sec'+ str(i+1) +'_Lat_LT_Z']>3 and event_record['Sec'+ str(i+1) +'_Lat_LT_Z']<=6:
                    print('Lateral Force: ' + Diag_Lat_LT[1])
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_LT_Z']/3*2
                elif event_record['Sec'+ str(i+1) +'_Lat_LT_Z']>6:
                    print('Lateral Force: ' + Diag_Lat_LT[2])
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_LT_Z']/3*3
            elif event_record['Type']=='RTT':
                if event_record['Sec'+ str(i+1) +'_Lat_RT_Z']>=-3:
                    print('Lateral Force: ' + Diag_Lat_RT[0])
                elif event_record['Sec'+ str(i+1) +'_Lat_RT_Z']<-3 and event_record['Sec'+ str(i+1) +'_Lat_RT_Z']>=-6:
                    print('Lateral Force: ' + Diag_Lat_RT[1])
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_RT_Z']/(-3)*2
                elif event_record['Sec'+ str(i+1) +'_Lat_RT_Z']<-6:
                    print('Lateral Force: ' + Diag_Lat_RT[2])
                    score = score + event_record['Sec'+ str(i+1) +'_Lat_RT_Z']/(-3)*3
    elif event_record['Type']=='LCR' or event_record['Type']=='LCL':    
        if event_record['Type']=='LCR':
            print('Lane change to right at a speed of ' + str(((event_record['Sec1_Speed(km/hr)']+event_record['Sec2_Speed(km/hr)'])/2).round(0)) + ' km/hr.')
        elif event_record['Type']=='LCL':
            print('Lane change to left at a speed of ' + str(((event_record['Sec1_Speed(km/hr)']+event_record['Sec2_Speed(km/hr)'])/2).round(0)) + ' km/hr.')
        #Acceleration/Deceleration 
        LC_Lon = np.max([event_record['Sec1_Lon_Acc_Z'], -1*event_record['Sec1_Lon_Dec_Z'], event_record['Sec2_Lon_Acc_Z'], -1*event_record['Sec2_Lon_Dec_Z']])
        if event_record['Start_Spd(km/h)']<event_record['End_Spd(km/h)']:
            if LC_Lon<=3:
                print('Acceleration: ' + Diag_Lon[0])
            elif LC_Lon>3 and LC_Lon<=6:
                print('Acceleration: ' + Diag_Lon[1])
                score = score + LC_Lon/3*2
            elif LC_Lon>6:
                print('Acceleration: ' + Diag_Lon[2])
                score = score + LC_Lon/3*3
        elif event_record['Start_Spd(km/h)']>event_record['End_Spd(km/h)']:
            if LC_Lon<=3:
                print('Deceleration: ' + Diag_Lon[0])
            elif LC_Lon>3 and LC_Lon<=6:
                print('Deceleration: ' + Diag_Lon[1])
                score = score + LC_Lon/3*2
            elif LC_Lon>6:
                print('Deceleration: ' + Diag_Lon[2])
                score = score + LC_Lon/3*3
        #Lateral Force        
        LC_Lat = np.max([event_record['Sec1_Lat_LT_Z'], -1*event_record['Sec1_Lat_RT_Z'], event_record['Sec2_Lat_LT_Z'], -1*event_record['Sec2_Lat_RT_Z']])     
        if LC_Lat<=3:
            print('You were changing lanes ' + Diag_Lane[0])
        elif LC_Lat>3 and LC_Lat<=6:
            print('You were changing lanes ' + Diag_Lane[1])
            score = score + LC_Lat/3*2
        elif LC_Lat>6:
            print('You were changing lanes ' + Diag_Lane[2])  
            score = score + LC_Lat/3*3
            
    print ('Your score is %s' % score.round(0))
    
    return score.round(0)
  
    
########################################################################################################
######  4. Function to make a query on status and display detailed analysis                       ######
######################################################################################################## 

def Status_Query(status_record):
         
    beg_time = datetime.fromtimestamp(status_record['Start_Timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    duration = status_record['Duration(s)']
    ave_spd = round((status_record['Start_Spd(km/h)']+status_record['End_Spd(km/h)'])/2,0)
    if status_record['Acceleration(m/s2)']<(-0.15):
        spd_status = 'Slow Down'
    elif status_record['Acceleration(m/s2)']>(0.15):
        spd_status = 'Speed Up'
    else:
        spd_status = 'Uniform Speed'
        
    #Print information
    if status_record['Type']=='CRS':   
        print('Cruise Straight Profile')
    elif status_record['Type']=='CRR': 
        print('Cruise to Right Profile')
    elif status_record['Type']=='CRL': 
        print('Cruise to Left Profile')
    elif status_record['Type']=='PRK': 
        print('Stop Profile')
        
    print('Beginning Time: %s' % beg_time)
    print('Duration: %s seconds' % duration)
    print('Average Speed: %s km/hr' % ave_spd)
    print(spd_status)
    
    return 0
    
########################################################################################################
######  5. Function for sequence analysis on status                                               ######
########################################################################################################    

def Sequence_Analysis(df_status, driver_id):
        
    statusLen = df_status.shape[0]
    duration = df_status['End_Timestamp'].iloc[statusLen-1] - df_status['Start_Timestamp'].iloc[0]
    spd_change = np.zeros(statusLen)  
    
    for i in range (statusLen):
        if i==0:
            spd_change[i]=0
        else:
            if (df_status['Acceleration(m/s2)'].iloc[i]>=0) and (df_status['Acceleration(m/s2)'].iloc[i-1]<0):
                spd_change[i] = 1
            elif (df_status['Acceleration(m/s2)'].iloc[i]>=0) and (df_status['Acceleration(m/s2)'].iloc[i-1]>=0):
                spd_change[i] = 2  
            elif (df_status['Acceleration(m/s2)'].iloc[i]<0) and (df_status['Acceleration(m/s2)'].iloc[i-1]<0):
                spd_change[i] = 3
            elif (df_status['Acceleration(m/s2)'].iloc[i]<0) and (df_status['Acceleration(m/s2)'].iloc[i-1]>=0):
                spd_change[i] = 4    
            
    df_seq = pd.DataFrame(np.nan, index=np.arange(1), columns=['Driver_ID','Duration(s)','Updown','Upup','Downdown','Downup','acc_ave',\
                             'acc_std','dec_ave','dec_std'])
    
    df_seq.iloc[0, df_seq.columns.get_loc('Driver_ID')] = driver_id
    df_seq.iloc[0, df_seq.columns.get_loc('Duration(s)')] = duration
    df_seq.iloc[0, df_seq.columns.get_loc('Updown')] = sum(num==1 for num in spd_change)
    df_seq.iloc[0, df_seq.columns.get_loc('Upup')] = sum(num==2 for num in spd_change)
    df_seq.iloc[0, df_seq.columns.get_loc('Downdown')] = sum(num==3 for num in spd_change)
    df_seq.iloc[0, df_seq.columns.get_loc('Downup')] = sum(num==4 for num in spd_change)
    df_seq.iloc[0, df_seq.columns.get_loc('acc_ave')] = df_status['Acceleration(m/s2)'].where(df_status['Acceleration(m/s2)']>=0).mean()
    df_seq.iloc[0, df_seq.columns.get_loc('acc_std')] = df_status['Acceleration(m/s2)'].where(df_status['Acceleration(m/s2)']>=0).std()
    df_seq.iloc[0, df_seq.columns.get_loc('dec_ave')] = df_status['Acceleration(m/s2)'].where(df_status['Acceleration(m/s2)']<0).mean()
    df_seq.iloc[0, df_seq.columns.get_loc('dec_std')] = df_status['Acceleration(m/s2)'].where(df_status['Acceleration(m/s2)']<0).std()
    
    return df_seq

########################################################################################################
######  6. Function for event analysis                                                            ######
######################################################################################################## 

def Event_Analysis(df_tot_eva, driver_id):
    
    df_evtana = pd.DataFrame(np.nan, index=np.arange(4), columns=['Driver_ID','Type','Ave_Duration',\
                             'Sec1_Lon_Acc_Z_ave','Sec1_Lon_Acc_Z_std','Sec1_Lon_Dec_Z_ave','Sec1_Lon_Dec_Z_std',\
                             'Sec1_Lat_LT_Z_ave','Sec1_Lat_LT_Z_std','Sec1_Lat_RT_Z_ave','Sec1_Lat_RT_Z_std',\
                             'Sec2_Lon_Acc_Z_ave','Sec2_Lon_Acc_Z_std','Sec2_Lon_Dec_Z_ave','Sec2_Lon_Dec_Z_std',\
                             'Sec2_Lat_LT_Z_ave','Sec2_Lat_LT_Z_std','Sec2_Lat_RT_Z_ave','Sec2_Lat_RT_Z_std',\
                             'Sec3_Lon_Acc_Z_ave','Sec3_Lon_Acc_Z_std','Sec3_Lon_Dec_Z_ave','Sec3_Lon_Dec_Z_std',\
                             'Sec3_Lat_LT_Z_ave','Sec3_Lat_LT_Z_std','Sec3_Lat_RT_Z_ave','Sec3_Lat_RT_Z_std'])
    
    evtType = ('RTT','LTT','LCR','LCL')

                
    for i in range(4):
        df_evtana.iloc[i, df_evtana.columns.get_loc('Driver_ID')] = driver_id
        df_evtana.iloc[i, df_evtana.columns.get_loc('Type')] = evtType[i]
        
        df_evtana.iloc[i, df_evtana.columns.get_loc('Ave_Duration')] = df_tot_eva['Duration(s)'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lon_Acc_Z_ave')] = df_tot_eva['Sec1_Lon_Acc_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lon_Acc_Z_std')] = df_tot_eva['Sec1_Lon_Acc_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lon_Dec_Z_ave')] = df_tot_eva['Sec1_Lon_Dec_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lon_Dec_Z_std')] = df_tot_eva['Sec1_Lon_Dec_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lat_LT_Z_ave')] = df_tot_eva['Sec1_Lat_LT_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lat_LT_Z_std')] = df_tot_eva['Sec1_Lat_LT_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lat_RT_Z_ave')] = df_tot_eva['Sec1_Lat_RT_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec1_Lat_RT_Z_std')] = df_tot_eva['Sec1_Lat_RT_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lon_Acc_Z_ave')] = df_tot_eva['Sec2_Lon_Acc_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lon_Acc_Z_std')] = df_tot_eva['Sec2_Lon_Acc_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lon_Dec_Z_ave')] = df_tot_eva['Sec2_Lon_Dec_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lon_Dec_Z_std')] = df_tot_eva['Sec2_Lon_Dec_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lat_LT_Z_ave')] = df_tot_eva['Sec2_Lat_LT_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lat_LT_Z_std')] = df_tot_eva['Sec2_Lat_LT_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lat_RT_Z_ave')] = df_tot_eva['Sec2_Lat_RT_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec2_Lat_RT_Z_std')] = df_tot_eva['Sec2_Lat_RT_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lon_Acc_Z_ave')] = df_tot_eva['Sec3_Lon_Acc_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lon_Acc_Z_std')] = df_tot_eva['Sec3_Lon_Acc_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lon_Dec_Z_ave')] = df_tot_eva['Sec3_Lon_Dec_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lon_Dec_Z_std')] = df_tot_eva['Sec3_Lon_Dec_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lat_LT_Z_ave')] = df_tot_eva['Sec3_Lat_LT_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lat_LT_Z_std')] = df_tot_eva['Sec3_Lat_LT_Z'].where(df_tot_eva['Type']==evtType[i]).std()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lat_RT_Z_ave')] = df_tot_eva['Sec3_Lat_RT_Z'].where(df_tot_eva['Type']==evtType[i]).mean()
        df_evtana.iloc[i, df_evtana.columns.get_loc('Sec3_Lat_RT_Z_std')] = df_tot_eva['Sec3_Lat_RT_Z'].where(df_tot_eva['Type']==evtType[i]).std()
    
    return df_evtana








