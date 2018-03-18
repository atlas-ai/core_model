 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:02:27 2017

@author: SeanXinZhou
"""

import pandas as pd
import numpy as np
import distributions


def thresholding_algo(x, lag, threshold, influence, avg, std):
    """ Use fixed thresholds as boundaries to generate signals of patterns

    :param x: input data
    :param lag: previous data points that are used to construct boundaries
    :param threshold: parameter for boundaries
    :param influence: numerical influence of previous data points
    :param avg: average of data points for initialisation
    :param std: standard deviation of data points for initialisation
    :return: dataframe that contains signals of patterns
    """

    y = x.copy(deep=True)
    sigs = pd.DataFrame(np.nan, index=y.index, columns=['Signal', 'Filter_Mean', 'Filter_Std'])
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

        sigs['Signal'] = signals
        sigs['Filter_Mean'] = avgFilter
        sigs['Filter_Std'] = stdFilter

    return sigs


def transformation_algo(signals, samp_rate):
    """ Transform signals to identifiable patterns

    :param signals:  signals of patterns
    :param samp_rate: sampling rate of data
    :return: transformation of signals
    """
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
            # 3*samp_rate: adding 3 seconds on both ends of transformed signals to capture maximum patterns
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
    """ Segment transformed signals through aggregating patterns close to each other

    :param trans:  transformed signals
    :return: segments of signals
    """
    
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


def evt_det_algo(rot_z, lat, long, alt, crs, spd, param, samp_rate, prob_thr, seg_code):
    """ Detect movements/events based on segment codes (for a singal segment)

    :param rot_z:  roatation rate of gyroscope for z-axis
    :param lat: latitude
    :param long: longitude
    :param alt: altitude
    :param crs: course
    :param spd: speed in km/hr
    :param param: coefficients for detection model
    :param samp_rate: sampling rate of data
    :param prob_thr: probability threshold for detection model
    :param seg_code: output of segmentation algorithm
    :return: detected events
    """
        
    dataLen = rot_z.shape[0]
    dataPoints = 20  # The number of data points to detect an event
    scanStep = int(samp_rate // 10)  # 1/10 of a second
    rotz_threshold = 0.03 # Enforce rotation rates close to 0 at both ends of patterns

    # Create empty data frame to store event data (RTT, LTT, LCR, LCL)
    df_evt = pd.DataFrame(np.nan, index=np.arange(1000), columns=['type', 'prob', 'd', 's_utc', 'e_utc', 'event_acc',\
                            's_spd', 'e_spd','s_crs', 'e_crs', 's_lat', 'e_lat', 's_long', 'e_long', 's_alt','e_alt'])
    evt_no = 1

    # Initialise scan parameters and probability threshold
    if seg_code == 1 or seg_code == 2:
        beg_window = int(5 * (samp_rate // dataPoints))
        end_window = int(18 * (samp_rate // dataPoints))
        num_of_window = 14
    elif seg_code == 3 or seg_code == 4:
        beg_window = int(2 * (samp_rate // dataPoints))
        end_window = int(8 * (samp_rate // dataPoints))
        num_of_window = 7
        
    #UPPER LOOP: Run detection model with different window sizes
    for stepSize in np.linspace(beg_window, end_window, num=num_of_window):

        windowSize = dataPoints * stepSize.astype(int)
        pre_evt = windowSize
        evt_max_prob = 0.0
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

        #BOTTOM LOOP: Move windows by 1/10 of second
        for i in range(0, dataLen - windowSize, scanStep):

            # Create an empty array to hold independent variables
            dataVar = np.empty(dataPoints * 3 + 1)
            dataVar[0] = 1.0

            # Extract values of key data points for a window segment
            for j in range(1, dataPoints + 1):
                idx = (i + (j - 1) * stepSize).astype(int)
                dataVar[j] = rot_z.iloc[idx] #Roatation rate w.r.t z-axis
                dataVar[j + dataPoints] = crs.iloc[idx] - crs.iloc[idx - 1] #Difference in course
                dataVar[j + 2 * dataPoints] = rot_z.index.values[idx] #Timestamps

            # Rotation w.r.t. z-axis must be close to zero to indicate the beginning and end of a event
            rotz_beg = dataVar[1]
            rotz_end = dataVar[20]

            # Calculate probability for data segment
            evt_prob = distributions.predict_prob_sigmoid(dataVar[0:2 * dataPoints + 1], param)

            # Identify events with pre-defined criteria
            if ((evt_prob >= prob_thr) and (np.abs(rotz_beg) <= rotz_threshold) and \
                        (np.abs(rotz_end) <= rotz_threshold)):
                # Check whether the detected event overlaps with previous event with the same window size
                if i <= pre_evt:
                    # Loop to record maximum probability
                    if evt_prob >= evt_max_prob:
                        evt_max_prob = evt_prob
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
                    # Data entry when scan step changes
                    if seg_code == 1:
                        #df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'RTT'
                        if ((end_crs-beg_crs)<-1.*(5/6)*3.14) and ((end_crs-beg_crs)>-1.*(7/6)*3.14):
                            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'utn'
                        else:
                            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'rtt'
                    elif seg_code == 2:
                        # When the change in course lies between -7/6pi and -5/6pi, classify the left turn as a u-turn.
                        if ((end_crs-beg_crs)<-1.*(5/6)*3.14) and ((end_crs-beg_crs)>-1.*(7/6)*3.14):
                            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'utn'
                        else:
                            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'ltt'
                    elif seg_code == 3:
                        df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'lcr'
                    elif seg_code == 4:
                        df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'lcl'
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_utc')] = beg_utc
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_utc')] = end_utc
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('d')] = (end_utc - beg_utc)
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_lat')] = beg_lat
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_lat')] = end_lat
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_long')] = beg_long
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_long')] = end_long
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_alt')] = beg_alt
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_alt')] = end_alt
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_spd')] = beg_spd
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_spd')] = end_spd
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_crs')] = beg_crs
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_crs')] = end_crs
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('prob')] = evt_max_prob
                    evt_no += 1

                    # Set values for a new event
                    evt_max_prob = evt_prob
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
                    pre_evt = i + windowSize

        # Data entry when time window changes
        if evt_max_prob != 0.0:
            if seg_code == 1:
                #df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'RTT'
                if ((end_crs-beg_crs)<-1.*(5/6)*3.14) and ((end_crs-beg_crs)>-1.*(7/6)*3.14):
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'utn'
                else:
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'rtt'
            elif seg_code == 2:
                if ((end_crs-beg_crs)<-1.*(5/6)*3.14) and ((end_crs-beg_crs)>-1.*(7/6)*3.14):
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'utn'
                else:
                    df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'ltt'
            elif seg_code == 3:
                df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'lcr'
            elif seg_code == 4:
                df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('type')] = 'lcl'
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_utc')] = beg_utc
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_utc')] = end_utc
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('d')] = (end_utc - beg_utc)
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_lat')] = beg_lat
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_lat')] = end_lat
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_long')] = beg_long
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_long')] = end_long
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_alt')] = beg_alt
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_alt')] = end_alt
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_spd')] = beg_spd
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_spd')] = end_spd
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('s_crs')] = beg_crs
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('e_crs')] = end_crs
            df_evt.iloc[evt_no - 1, df_evt.columns.get_loc('prob')] = evt_max_prob
            evt_no += 1

    df_evt = df_evt[df_evt['prob'] > prob_thr]
    df_evt = df_evt.sort_values(['type', 'e_utc', 'prob'], ascending=[True, True, False])
    df_evt = df_evt.reset_index(drop=True)
    df_evt['d'] = df_evt['d'] / 1000000000
    df_evt['s_utc'] = pd.to_datetime(df_evt['s_utc'] / 1000000000, unit='s')
    df_evt['e_utc'] = pd.to_datetime(df_evt['e_utc'] / 1000000000, unit='s')
    df_evt['event_acc'] = (df_evt['e_spd'] - df_evt['s_spd']) / df_evt['d'] / 3.6 / 9.8

    return df_evt


def remove_evt_duplicates(df_evt):
    """ remove duplicates and summarise event detection results

    :param df_event: preliminary dataframe for events   
    :return: detected events
    """
    
    if df_evt.empty==False:
        
        #Selecting the event with highest probability given the same type of events that overlap in time
        evtLen = df_evt.shape[0]
        beg_utc = df_evt['s_utc'].iloc[0]
        end_utc = df_evt['e_utc'].iloc[0]
        type_idx = df_evt['type'].iloc[0]
        df_evt['overlap'] = 0
        overlap_idx = 1        
        for i in range(evtLen):
            if df_evt['type'].iloc[i]!=type_idx: 
                beg_utc = df_evt['s_utc'].iloc[i]
                end_utc = df_evt['e_utc'].iloc[i]
                type_idx = df_evt['type'].iloc[i]
                overlap_idx += 1
            if df_evt['s_utc'].iloc[i] > end_utc: 
                end_utc = df_evt['e_utc'].iloc[i]
                overlap_idx += 1
                df_evt.iloc[i, df_evt.columns.get_loc('overlap')] = overlap_idx
            else:
                if (((end_utc-df_evt['s_utc'].iloc[i])/(end_utc-beg_utc)>=1/3) or\
                    ((end_utc-df_evt['s_utc'].iloc[i])/(df_evt['e_utc'].iloc[i]-df_evt['s_utc'].iloc[i])>=1/3)):
                    df_evt.iloc[i, df_evt.columns.get_loc('overlap')] = overlap_idx
                else:
                    end_utc = df_evt['e_utc'].iloc[i]
                    overlap_idx += 1
                    df_evt.iloc[i, df_evt.columns.get_loc('overlap')] = overlap_idx                                                  
                
        df_evt = df_evt.loc[df_evt.reset_index().groupby(['overlap'])['prob'].idxmax()]
        df_evt = df_evt.reset_index(drop=True) 
        df_evt = df_evt.drop('overlap', axis=1)
        
        #Selecting the event with highest probability given different types of events that overlap in time
        df_evt = df_evt.sort_values(['e_utc', 'prob'], ascending=[True, False])  
        df_evt = df_evt.reset_index(drop=True)
        evtLen = df_evt.shape[0]
        beg_utc = df_evt['s_utc'].iloc[0]
        end_utc = df_evt['e_utc'].iloc[0]
        type_idx = df_evt['type'].iloc[0]
        df_evt['overlap'] = 0
        overlap_idx = 1
        for i in range(evtLen):
            if df_evt['s_utc'].iloc[i] > end_utc: 
                end_utc = df_evt['e_utc'].iloc[i]
                overlap_idx += 1
                df_evt.iloc[i, df_evt.columns.get_loc('overlap')] = overlap_idx
            else:
                if (((end_utc-df_evt['s_utc'].iloc[i])/(end_utc-beg_utc)>=1/3) or\
                    ((end_utc-df_evt['s_utc'].iloc[i])/(df_evt['e_utc'].iloc[i]-df_evt['s_utc'].iloc[i])>=1/3)):
                    df_evt.iloc[i, df_evt.columns.get_loc('overlap')] = overlap_idx
                else:
                    end_utc = df_evt['e_utc'].iloc[i]
                    overlap_idx += 1
                    df_evt.iloc[i, df_evt.columns.get_loc('overlap')] = overlap_idx  
              
        df_evt = df_evt.loc[df_evt.reset_index().groupby(['overlap'])['prob'].idxmax()]
        df_evt = df_evt.reset_index(drop=True) 
        df_evt = df_evt.drop('overlap', axis=1)
        
    return df_evt    

def event_detection(rot_z, lat, long, alt, crs, spd, evt_param, samp_rate, tn_thr, lc_thr):
    """ Encapsulate thresholding_algo, transformation_algo, segmentation_algo, evt_det_algo and remove_evt_duplicates

    :param rot_z:  roatation rate of gyroscope for z-axis
    :param lat: latitude
    :param long: longitude
    :param alt: altitude
    :param crs: course
    :param spd: speed in km/hr
    :param evt_param: coefficients for event detection model
    :param samp_rate: sampling rate of data
    :param tn_thr: probability threshold for turns
    :param lc_thr: probability threshold for lane changes
    :return: detected events
    """
    
    df_event = pd.DataFrame(np.nan, index=np.arange(0), columns=['type', 'prob', 'd', 's_utc', 'e_utc', 'event_acc',\
                          's_spd', 'e_spd','s_crs', 'e_crs', 's_lat', 'e_lat', 's_long', 'e_long', 's_alt', 'e_alt'])
    
    # Step 1: Pre-screen to obtain signal segments
    rot_z_avg = rot_z.where(np.abs(rot_z) <= 0.02).mean()
    rot_z_std = rot_z.where(np.abs(rot_z) <= 0.02).std()
    sigs = thresholding_algo(rot_z, lag=5, threshold=6, influence=0.01, avg=rot_z_avg, std=rot_z_std)
    trans = transformation_algo(sigs, samp_rate)
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
                             evt_param[int(sc - 1)], samp_rate, prob_thr=tn_thr, seg_code=sc)
                df_event = df_event.append(df)
            elif sc == 2.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, prob_thr=tn_thr, seg_code=sc)
                df_event = df_event.append(df)
            elif sc == 3.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, prob_thr=lc_thr, seg_code=sc)
                df_event = df_event.append(df)
            elif sc == 4.:
                df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                             evt_param[int(sc - 1)], samp_rate, prob_thr=lc_thr, seg_code=sc)
                df_event = df_event.append(df)
            elif sc == 5.:
                for j in range(4):
                    if j == 0 or j == 1:
                        evt_thr = tn_thr
                    elif j == 2 or j == 3:
                        evt_thr = lc_thr
                    df = evt_det_algo(rot_z_seg, lat_seg, long_seg, alt_seg, crs_seg, spd_seg, \
                                 evt_param[j], samp_rate, prob_thr=evt_thr, seg_code=(j + 1))
                    df_event = df_event.append(df)

        df_event = df_event.reset_index(drop=True)
        df_event = remove_evt_duplicates(df_event)

    return df_event


def ex_acc_det_algo(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr):
    """ detect the excess acceleration or deceleration 
    
    :param acc_x: longitudinal force of a vehicle
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param crs: course in radians
    :param spd: speed of a vehicle in km/hr
    :param acc_param: coefficients of detection model
    :param samp_rate: sampling rate of raw data
    :param acc_thr: threshold of z-score that acceleration breaches
    :return: data frame to summarise the occasions of excess acceleration
    """   

    df_acc = pd.DataFrame(np.nan, index=np.arange(10000), columns=['type','prob','d',\
                              's_utc','e_utc','event_acc','s_spd','e_spd','s_crs','e_crs',\
                              's_lat','e_lat','s_long','e_long','s_alt','e_alt','duplicate'])
    acc_no = 0
    max_acc = 0.0
    max_dec = 0.0             
    dataLen = len(acc_x)
    scanStep = int(samp_rate//10) #1/10 of a second  
    
    for i in range(1, dataLen, scanStep):       
        if acc_x[i] > (acc_param['acc_ave'][0] + acc_thr*np.sqrt(acc_param['acc_var'][0])):
            if (acc_x[i]>=max_acc) and (acc_x[i]>=acc_x[i-1]):
                max_acc = acc_x[i]
                max_utc = acc_x.index[i] 
            elif (acc_x[i]<max_acc) and (acc_x[i]<acc_x[i-1]):
                df_acc.iloc[acc_no, df_acc.columns.get_loc('type')] = 'exa'
                df_acc.iloc[acc_no, df_acc.columns.get_loc('e_utc')] = max_utc
                df_acc.iloc[acc_no, df_acc.columns.get_loc('e_spd')] = spd[i-1]
                df_acc.iloc[acc_no, df_acc.columns.get_loc('event_acc')] = max_acc
                acc_no += 1
                max_acc = 0

        elif acc_x[i] < (acc_param['dec_ave'][0] - acc_thr*np.sqrt(acc_param['dec_var'][0])):
            if (acc_x[i]<=max_dec) and (acc_x[i]<=acc_x[i-1]):
                max_dec = acc_x[i]
                max_utc = acc_x.index[i] 
            elif (acc_x[i]>max_dec) and (acc_x[i]>acc_x[i-1]):
                df_acc.iloc[acc_no, df_acc.columns.get_loc('type')] = 'exd'
                df_acc.iloc[acc_no, df_acc.columns.get_loc('e_utc')] = max_utc
                df_acc.iloc[acc_no, df_acc.columns.get_loc('e_spd')] = spd[i-1]
                df_acc.iloc[acc_no, df_acc.columns.get_loc('event_acc')] = max_dec
                acc_no += 1
                max_dec = 0
    
    df_acc = df_acc.dropna(how='all')
    
    return df_acc


def ex_acc_expansion(acc_x, lat, long, alt, crs, spd, df_acc, samp_rate):
    """Expand maximum point acceleration to period

    :param df_acc: preliminary dataframe for excess acceleration  
    :param samp_rate: sampling rate of data
    :return: detected excess accelerations as periods in time
    """  
    
    if df_acc.empty==False:       
        #Expand 2.5 seonds to both ends from a single point of maximum acceleration
        t_shift = int(3*samp_rate)
        accLen = df_acc.shape[0]
        
        for i in range(accLen):
            idx = acc_x.index.searchsorted(df_acc['e_utc'][i]) 
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

                df_acc.iloc[i, df_acc.columns.get_loc('s_utc')] = s_utc
                df_acc.iloc[i, df_acc.columns.get_loc('s_lat')] = s_lat
                df_acc.iloc[i, df_acc.columns.get_loc('s_long')] = s_long
                df_acc.iloc[i, df_acc.columns.get_loc('s_alt')] = s_alt               
                df_acc.iloc[i, df_acc.columns.get_loc('s_spd')] = s_spd
                df_acc.iloc[i, df_acc.columns.get_loc('s_crs')] = s_crs
                df_acc.iloc[i, df_acc.columns.get_loc('e_utc')] = e_utc
                df_acc.iloc[i, df_acc.columns.get_loc('e_lat')] = e_lat
                df_acc.iloc[i, df_acc.columns.get_loc('e_long')] = e_long
                df_acc.iloc[i, df_acc.columns.get_loc('e_alt')] = e_alt
                df_acc.iloc[i, df_acc.columns.get_loc('e_spd')] = e_spd
                df_acc.iloc[i, df_acc.columns.get_loc('e_crs')] = e_crs        
                duration = (e_utc-s_utc)/np.timedelta64(1, 's')
                df_acc.iloc[i, df_acc.columns.get_loc('d')] = duration
                if e_spd>=s_spd:
                    df_acc.iloc[i, df_acc.columns.get_loc('type')] = 'exa'
                else:
                    df_acc.iloc[i, df_acc.columns.get_loc('type')] = 'exd'
                df_acc.iloc[i, df_acc.columns.get_loc('event_acc')] = (e_spd-s_spd)/3.6/duration/9.8

        df_acc['prob']=1.0
        df_acc = df_acc[df_acc['d']>0]
        
    return df_acc


def remove_acc_duplicates(df_acc):    
    """ remove duplicates of excess acceleration

    :param df_acc: preliminary dataframe for excess acceleration   
    :return: detected excess accelerations as points in time
    """
    
    if df_acc.empty==False:
        
        df_acc['duplicate']=0
        accLen=df_acc.shape[0]
        df_acc.iloc[0,df_acc.columns.get_loc('duplicate')]=1
        overlap_indicator = 1
        df_acc['event_acc']=df_acc['event_acc'].apply(lambda x: -1*x if x<0 else x)
            
        for i in range(1, accLen):
            
            if df_acc['type'][i]==df_acc['type'][i-1]:
                if (df_acc['e_utc'][i]-df_acc['e_utc'][i-1])/np.timedelta64(1, 's')<=2:
                    df_acc.iloc[i,df_acc.columns.get_loc('duplicate')]=overlap_indicator
                else:
                    overlap_indicator += 1
                    df_acc.iloc[i,df_acc.columns.get_loc('duplicate')]=overlap_indicator
            else:
                overlap_indicator += 1
                df_acc.iloc[i,df_acc.columns.get_loc('duplicate')]=overlap_indicator
        
        df_acc = df_acc.loc[df_acc.groupby('duplicate')['event_acc'].idxmax()]        
        df_acc = df_acc.reset_index(drop=True)
        
        accLen = df_acc.shape[0]
        for i in range(accLen):
            if df_acc['type'][i]=='exd':
                temp_max_dec = df_acc['event_acc'][i]
                df_acc.iloc[i,df_acc.columns.get_loc('event_acc')]=-1*temp_max_dec
        
        df_acc = df_acc.drop('duplicate',axis=1)
        
    return df_acc


def ex_acc_detection(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr):
    """ Encapsulate ex_acc_det_algo, remove_acc_duplicates and ex_acc_expansion 
    
    :param acc_x: longitudinal force of a vehicle
    :param lat: gps latitude in degree
    :param long: gps longitude in degree
    :param alt: gps altitude in metre
    :param crs: course in radians
    :param spd: speed of a vehicle in km/hr
    :param acc_param: coefficients of detection model
    :param samp_rate: sampling rate of raw data
    :param acc_thr: threshold of z-score that acceleration breaches
    :return: data frame to summarise the occasions of excess acceleration
    """   
    
    df_acc = ex_acc_det_algo(acc_x, lat, long, alt, crs, spd, acc_param, samp_rate, acc_thr)    
    df_acc = remove_acc_duplicates(df_acc)
    df_acc = ex_acc_expansion(acc_x, lat, long, alt, crs, spd, df_acc, samp_rate)
    df_acc = df_acc.reset_index(drop=True)
    
    return df_acc
    

          