# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:27:37 2017

@author: seanx
"""

import pandas as pd
import numpy as np
import timeit
import Function_Input as fin
import Function_Filter as ffi
import Function_Frame as ffc
import Function_Detection as fdet
import Function_Evaluation as feva
import Function_Query as fque
import Function_Plot as fplt

beg_time = timeit.default_timer()

###########################################
#     Initialise Final Result Table       #
###########################################        

features=['Type','Duration(s)','Start_Index','End_Index','Start_Timestamp','End_Timestamp',\
          'Start_Spd(km/h)','End_Spd(km/h)','Acceleration(m/s2)','Start_Course','End_Course','Probability',\
          'Sec1_Speed(km/hr)','Sec1_Lon_Acc_Z','Sec1_Lon_Dec_Z','Sec1_Lat_LT_Z','Sec1_Lat_RT_Z',\
          'Sec2_Speed(km/hr)','Sec2_Lon_Acc_Z','Sec2_Lon_Dec_Z','Sec2_Lat_LT_Z','Sec2_Lat_RT_Z',\
          'Sec3_Speed(km/hr)','Sec3_Lon_Acc_Z','Sec3_Lon_Dec_Z','Sec3_Lat_LT_Z','Sec3_Lat_RT_Z']

df_tot_eva = pd.DataFrame(np.NaN, index=np.arange(1), columns=features)
df_tot_eva = df_tot_eva.dropna()

###########################################
#               Input Data                #
########################################### 

#Read data from Excel
df = fin.input_IMU('20170222_07_48_55_Uber.xlsx', '20170222_07_48_55_Uber', 'GPS').iloc[0:8000]

#Apply filter (parameters: DataFrame, Moving Average Window Size)
df_ma = ffi.MA_Filter(df, 100)

#Convert from device-frame to geo-frame (parameters: DataFrame, Sampling Rate)
df_fc = ffc.Frame_Conversion(df_ma, 100)

#Event Detection model
df_event = fdet.Event_Detection(df_fc)

#Evaluate events
df_eva = feva.Event_Evaluation(df_fc, df_event)

#Summarise event table
df_tot_eva = df_tot_eva.append(df_eva, ignore_index=True)   
df_tot_eva = df_tot_eva[features]
df_tot_eva = df_tot_eva.reset_index(drop=True)  

#Remove duplication from event evaluation table
evaLen = df_tot_eva.shape[0]
df_tot_eva['duplicates'] = 0
for i in range(1, evaLen):
    if df_tot_eva['Start_Timestamp'].iloc[i] == df_tot_eva['Start_Timestamp'].iloc[i-1] and df_tot_eva['End_Timestamp'].iloc[i] == df_tot_eva['End_Timestamp'].iloc[i-1]:
        df_tot_eva['duplicates'].iloc[i]=1        
df_tot_eva = df_tot_eva.loc[df_tot_eva['duplicates']==0]
df_tot_eva = df_tot_eva[features]

#Status Detection
df_status = fdet.Status_Detection(df_fc, df_tot_eva)

#Detect emergency break (parameters: DataFrame)
df_break = fdet.Breaking_Detection(df_fc)

#Summarise all information into one DataFrame
df_sum, tot_score = fque.Data_Summary(df_tot_eva, df_status, df_break)  
print(tot_score)
