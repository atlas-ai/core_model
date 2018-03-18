#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:47:32 2018

@author: Sean_Xin_Zhou
"""

import numpy as np

def acc_adjustment(df, n_smooth, g=9.8):
    """adjust accelerations of x and y axes 
    
    :param df: input dataframe after frame conversion
    :return: adjustment in accelerations
    """
    acc_x_diff = np.exp(np.abs(df['acc_x']-df['acc_x_gps']))
    acc_x_fac = 1/acc_x_diff
    acc_x = ((1-acc_x_fac)*df['acc_x'] + acc_x_fac*df['acc_x_gps']).rolling(n_smooth).mean().bfill()
    
    acc_y_est = df['r_rate_z']*df['speed']/g
    acc_y_diff = np.exp(np.abs(df['acc_y']-acc_y_est))
    fac_y_fac = 1/acc_y_diff
    acc_y = ((1-fac_y_fac)*df['acc_y'] + fac_y_fac*acc_y_est).rolling(n_smooth).mean().bfill()
    
    rot_z = df['r_rate_z'].rolling(n_smooth).mean().bfill()
    lat = df['lat'].rolling(n_smooth).mean().bfill()
    long = df['long'].rolling(n_smooth).mean().bfill()
    alt = df['alt'].rolling(n_smooth).mean().bfill()
    crs = df['course'].rolling(n_smooth).mean().bfill()
    spd = df['speed'].rolling(n_smooth).mean().bfill()*3.6
    acc_x_gps = df['acc_x_gps'].rolling(n_smooth).mean().bfill()
    acc_y_gps = df['acc_y_gps'].rolling(n_smooth).mean().bfill()
    
    return acc_x, acc_y, rot_z, lat, long, alt, crs, spd, acc_x_gps, acc_y_gps


