# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:21:04 2017

@author: seanx
"""

import pandas as pd
import numpy as np
import Function_Math as fmat
import timeit
import quaternion as qtr


def course_to_frame(course):
    """ provides the geoframe of a car given the gps course angle

    We assumes that the car is standing on its wheels.


    :param course: clockwise angle from the north of the gps course in radian
    :return: frame as a quaternion
    """
    # we directly use the quaternion definition rotation
    # with a rotation in the z direction (cf geoframe definition)
    # and an angle given by the gps course
    # r = exp(course/2*k)
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    return np.exp(course/2 * qtr.quaternion(0, 0, 0, 1))


def ypr_to_frame(yawn, pitch, roll):
    """    Generate unit quaternion frame from yawn, pitch and roll

    yawn -> rotation around z axis
    pitch -> rotation around x axis
    roll -> rotation around y axis

    the rotation are applied in this order:

     yawn -> pitch -> roll

    the formula used is `exp(yawn*z/2) * exp(pitch*x/2) * exp(roll*y/2)`


    :param yawn: alpha in radian
    :param pitch: beta in radian
    :param roll: gamma in radian
    :return: frame as a quaternion
    """

    r_yawn = np.exp(yawn/2 * qtr.quaternion(0, 0, 0, 1))
    r_pitch = np.exp(pitch/2 * qtr.quaternion(0, 1, 0, 0))
    r_roll = np.exp(roll/2 * qtr.quaternion(0, 0, 1, 0))
    res = r_yawn * r_pitch * r_roll
    return res


def rotate_vec(v_origin, v_dest):
    """ Generate quaternion rotating v_origin to v_dest

    This is not uniquely defined,
    we return the rotation axis providing the shorter angle

    This function is based on quaternion algebra only

    it uses the fact that when v_1 and v_2 are unit pure vector

    ```v_1 * v_2 = -cos(theta) + sin(theta) v_3```

    where v_3 is the unit rotation vector from v_1 to v_2
    and that the rotation vector from v_1 to v_2 is

     ```exp(theta/2 v_3)=  cos(theta/2) + sin(theta/2) v_3```



    :param v_origin: origin vector as quaternion
    :param v_dest: destination vector as quaternion
    :return: rotation quaternion
    """
    # other potential approach
    # http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors

    v_origin = v_origin / np.absolute(v_origin)
    v_dest = v_dest / np.absolute(v_dest)

    res = np.exp(0.5*np.log(- np.conjugate(v_origin * v_dest)))

    return res


def rotate_2vec(v_origin1, v_dest1, v_origin2, v_dest2):
    """ Generate quaternion rotating v_origin1 to v_dest1 and v_origin2 to v_dest2

    works if v_origin1 and v_origin2 are orthogonal and so are v_dest1 and v_dest2

    :param v_origin1: origin vector 1 q as quaternion
    :param v_dest1: destination vector 1 as quaternion
    :param v_origin2: origin vector 2 q as quaternion
    :param v_dest2: destination vector 2 as quaternion
    :return: rotation quaternion
    """

    r1 = rotate_vec(v_origin1, v_dest1)

    v_o2 = r1 * v_origin2 * np.conjugate(r1)

    r2 = rotate_vec(v_o2, v_dest2)

    res = r2 * r1

    return res


def mag_grav_to_frame(g, m):
    """ Provides the geoframe of the imu given magnetic field and gravity

    :param g: gravity as a quaternion
    :param m: magnetic field as a quaternion
    :return: frame as a quaternion
    """
    # gravity toward z in the geoframe
    y = fmat.cross(g, m)  # y direction in geoframe normal to g and m
    res = rotate_2vec(g, qtr.z, y, qtr.y)
    return res


def Frame_Conversion(df, freq):
    start = timeit.default_timer()

##############################################################################################################################
#                                                          Read DataFrame                                                    #
##############################################################################################################################

    #Attitude
    att = df['attitude_sum']
    pitch = df['attitude_pitch(radians)']
    roll = df['attitude_roll(radians)']
    yaw = df['attitude_yaw(radians)']
    
    #Rotation
    rotx = df['rotation_rate_x(radians/s)']
    roty = df['rotation_rate_y(radians/s)']
    rotz = df['rotation_rate_z(radians/s)']

    #Gravity
    gx = df['gravity_x(G)']
    gy = df['gravity_y(G)']
    gz = df['gravity_z(G)']

    #Acceleration
    ax = df['user_acc_x(G)']
    ay = df['user_acc_y(G)']
    az = df['user_acc_z(G)']

    #Magnetic Field
    mx = df['magnetic_field_x(microteslas)']
    my = df['magnetic_field_y(microteslas)']
    mz = df['magnetic_field_z(microteslas)']
    
    #GPS data
    lat = df['latitude(degree)']
    lon = df['longitude(degree)']
    alt = df['altitude(meter)']
    spd = df['speed(m/s)']
    course = df['course(degree)']
    
    #Convert course from degree to radian
    crs = course/180*np.pi
    crs_idx = df['course(degree)'].isnull()

    #Data Collection Frequency (100Hz)
    deltaT = 1/freq 
    
    #Data Length
    dataLen = df.shape[0]

##############################################################################################################################
#         Step 1: Building Static Conversion Matrix (Device-Frame to Geo-Frame) through Gravity and Magnetic Field           #
##############################################################################################################################

    #Normalising to get uG, uB, uE and uN
    uG = fmat.DataNorm(gx, gy, gz)
    uGx = uG['x']
    uGy = uG['y']
    uGz = uG['z']
    uB = fmat.DataNorm(mx, my, mz)
    #uBx = uB['x']
    #uBy = uB['y']
    #uBz = uB['z']
    E = np.cross(uG, uB)
    uE = fmat.DataNorm(E[:,0], E[:,1], E[:,2])
    uEx = uE['x']
    uEy = uE['y']
    uEz = uE['z']
    N = np.cross(uE, uG)
    uN = fmat.DataNorm(N[:,0], N[:,1], N[:,2])
    uNx = uN['x']
    uNy = uN['y']
    uNz = uN['z']

    #Construct static conversion matrix
    Static_Conversion_Matrix = np.asarray([[uNx, uNy, uNz], [uEx, uEy, uEz], [uGx, uGy, uGz]]).transpose()

##############################################################################################################################
#                        Step 2: Adding Dynamic Rotation into Conversion Matrix through Gyroscope                            #
##############################################################################################################################

    #Convert Euler angles to quaternions and integrate rotation rates to obtain changes in rotation
    q = fmat.EulerToQuaternion(pitch, roll, yaw)
    qdt = fmat.IntegrationOfAngularVelocity(rotx, roty, rotz, q, deltaT)
    Dynamic_Conversion_Matrix = fmat.QuaternionToRotMatrix(qdt)

    Conversion_Matrix = Static_Conversion_Matrix + Dynamic_Conversion_Matrix

    revisedrotation = np.zeros((dataLen,3))
    revisedacceleration = np.zeros((dataLen,3))
    revisedgravity = np.zeros((dataLen,3))
    revisedmagnetics = np.zeros((dataLen,3))
    for i in range(0, dataLen):        
        revisedrotation[i] = np.dot(np.asarray([rotx.iloc[i], roty.iloc[i], rotz.iloc[i]]), Conversion_Matrix[i])
        revisedacceleration[i] = np.dot(np.asarray([ax.iloc[i], ay.iloc[i], az.iloc[i]]), Conversion_Matrix[i])
        revisedgravity[i] = np.dot(np.asarray([gx.iloc[i], gy.iloc[i], gz.iloc[i]]), Conversion_Matrix[i])
        revisedmagnetics[i] = np.dot(np.asarray([mx.iloc[i], my.iloc[i], mz.iloc[i]]), Conversion_Matrix[i])
        
    REV_rot = pd.DataFrame(revisedrotation, index=df.index.values) 
    REV_rot.columns = ['rot_x','rot_y','rot_z']
    
    REV_acc = pd.DataFrame(revisedacceleration, index=df.index.values) 
    REV_acc.columns = ['acc_x','acc_y','acc_z']
    
    REV_gra = pd.DataFrame(revisedgravity, index=df.index.values) 
    REV_gra.columns = ['gra_x','gra_y','gra_z']
    
    REV_mag = pd.DataFrame(revisedmagnetics, index=df.index.values) 
    REV_mag.columns = ['mag_x','mag_y','mag_z']

##############################################################################################################################
#            Step 3: Determining driving direction and finalising longitudinal and lateral rotation/acceleration             #
##############################################################################################################################

    #Using GPS course as initial direction
    #Using z-axis rotation to determine changes in course
    delta_course = np.zeros(dataLen)
    ini_course = 0
    
    #Find initial course
    for i in range(0, dataLen):
        if crs_idx.iloc[i]==False:
            if crs.iloc[i]!=-1:
                ini_course = crs.iloc[i]
                break
    delta_course[0]=ini_course + 0.5*np.pi   #Half of pi is added due to smart phone identify West as 0 for z-axis
      
    #Update course when changing    
    for i in range(1, dataLen):
        delta_course[i] = delta_course[i-1] - (REV_rot['rot_z'].iloc[i])*deltaT

    fc_roty = (REV_rot['rot_y']*np.cos(delta_course)+REV_rot['rot_x']*np.sin(delta_course)).round(6)
    fc_rotx = (-REV_rot['rot_y']*np.sin(delta_course)+REV_rot['rot_x']*np.cos(delta_course)).round(6)
    fc_rotz = (REV_rot['rot_z']).round(6)

    fc_accy = (REV_acc['acc_y']*np.cos(delta_course)+REV_acc['acc_x']*np.sin(delta_course)).round(6)
    fc_accx = (-REV_acc['acc_y']*np.sin(delta_course)+REV_acc['acc_x']*np.cos(delta_course)).round(6)
    fc_accz = (REV_acc['acc_z']).round(6)
    
##############################################################################################################################
#                                                         Output DataFrame                                                   #
##############################################################################################################################
    
    features=['attitude_sum','attitude_pitch(radians)','attitude_roll(radians)','attitude_yaw(radians)','rotation_rate_x(radians/s)',\
             'rotation_rate_y(radians/s)','rotation_rate_z(radians/s)','gravity_x(G)','gravity_y(G)','gravity_z(G)','user_acc_x(G)',\
             'user_acc_y(G)','user_acc_z(G)','magnetic_field_x(microteslas)','magnetic_field_y(microteslas)','magnetic_field_z(microteslas)',\
             'latitude(degree)','longitude(degree)','altitude(meter)','speed(m/s)','course(degree)']
    df_fc = pd.concat([att, pitch, roll, yaw, fc_rotx, fc_roty, fc_rotz, gx, gy, gz,\
                       fc_accx, fc_accy, fc_accz, mx, my, mz, lat, lon, alt, spd, course], axis=1)   
    df_fc.columns = features
    
    #df_fc.to_csv('fc.csv', index_label='timestamp(unix)')

    stop = timeit.default_timer()
    print ("Frame Conversion Run Time: %s seconds \n" % round((stop - start),2))
    return df_fc

















