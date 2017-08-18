# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 08:59:21 2017

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
import Function_Summary as fsum
import Function_Plot as fplt

def main(file_name, IMU_sheet, GPS_sheet):
    beg_time = timeit.default_timer()
    rec_id = IMU_sheet

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
    df = fin.input_IMU(file_name, IMU_sheet, GPS_sheet)

    #Apply filter (parameters: DataFrame, Moving Average Window Size)
    df_ma = ffi.MA_Filter(df, 100)

    #Convert from device-frame to geo-frame (parameters: DataFrame, Sampling Rate)
    df_fc = ffc.Frame_Conversion(df_ma, 100)

###########################################
#                  Run All                #
########################################### 

    #Detect right turn, left turn, lane change to right and lane change to left (parameters: DataFrame)
    df_event = fdet.Event_Detection(df_fc)
    
    #Evaluate events
    df_tot_eva = feva.Event_Evaluation(df_fc, df_event)
        
###########################################
#      Status Detection and Summary       #
###########################################

    #Remove duplication from event evaluation table
    evaLen = df_tot_eva.shape[0]
    df_tot_eva['duplicates'] = 0
    for i in range(1, evaLen):
        if df_tot_eva['Start_Timestamp'].iloc[i] == df_tot_eva['Start_Timestamp'].iloc[i-1] and df_tot_eva['End_Timestamp'].iloc[i] == df_tot_eva['End_Timestamp'].iloc[i-1]:
            df_tot_eva['duplicates'].iloc[i]=1        
    df_tot_eva = df_tot_eva.loc[df_tot_eva['duplicates']==0]
    df_tot_eva = df_tot_eva[features]

    #Detect overall status of cruise and parking, in terms of sections and accelerations (parameters: DataFrame)
    df_status = fdet.Status_Detection(df_fc, df_tot_eva, 1)

    #Detect emergency break (parameters: DataFrame)
    df_brake = fdet.Braking_Detection(df_fc)

###########################################
#            Output DataFrame             #
###########################################

    #Summarise all information into one DataFrame
    df_sum, tot_score = fsum.Data_Summary(df_tot_eva, df_status, df_brake)  

    df_seq = fsum.Sequence_Analysis(df_status, rec_id)
    df_seq.to_csv((rec_id + '_seq.csv'), index=False)

    df_evtana = fsum.Event_Analysis(df_tot_eva, rec_id)
    df_evtana.to_csv((rec_id + '_evt.csv'), index=False)

    #fplt.Event_Plot(14, df_sum, df_fc)

    print('Total score for this trip is %s.\n' % tot_score)
    
    end_time = timeit.default_timer()
    print ("Total Run Time: %s seconds \n" % round((end_time - beg_time),2))

    return tot_score



