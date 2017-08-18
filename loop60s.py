"""
###########################################
#           Run Every 60 Seconds          #
########################################### 

#Simulate reading patterns (Do caculation every 60 seconds, with 15 seconds overlapping data.)
beg_rec = 1
end_rec = 6000
tot_rep = (df.shape[0]-6000)//4500+1

for i in range(tot_rep):    
    if i==(tot_rep-1):
        end_rec = df_fc.shape[0]-1

    df_segment = df_fc.iloc[beg_rec-1:end_rec]
    beg_rec += 4500
    end_rec += 4500
    
    ######################################
    #   Event Detection and Evaluation   #
    ######################################
    
    print('Event Detection & Evaluation Loop %s' % (i+1))
    
    #Since event detection is most time consuming process, it is calculated every 60 seconds.
    #Detect right turn, left turn, lane change to right and lane change to left (parameters: DataFrame)
    df_event = fdet.Event_Detection(df_segment)
    
    #Evaluate events
    df_eva = feva.Event_Evaluation(df_segment, df_event)
    
    #Summarise event table
    df_tot_eva = df_tot_eva.append(df_eva, ignore_index=True)   
    df_tot_eva = df_tot_eva[features]
    df_tot_eva = df_tot_eva.reset_index(drop=True)  
"""