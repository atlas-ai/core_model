#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:10:31 2017

@author: SeanXinZhou

Note: The coordinate system of a smartphone is different from that of an airplane.
      The x-axis of a smartphone represents pitch, whereas it is roll for an airplane.
      The y-axis of a smartphone represents roll, whereas it is pitch for an airplane.
      The z-axes of both bodies are the same. 
      The constructions of direction cosine matrix and quaternion are different between these two bodies.
"""

import pandas as pd
import numpy as np

#############################################################################################
#                        Euler Angles - Rotation Matrix - Quaternion                        #
#############################################################################################

#Euler Angles to Direction Cosine Matrix: Sequence ZXY(312) - Yaw Pitch Roll
#Rotation Sequence: Roll -> Pitch -> Yaw (Reverse Order)
def EulerToDCM(x, y, z):
    sinTheta = np.sin(x)
    cosTheta = np.cos(x)
    sinPhi = np.sin(y)
    cosPhi = np.cos(y)
    sinPsi = np.sin(z)
    cosPsi = np.cos(z)
    r11 =  cosPsi*cosPhi + sinPsi*sinTheta*sinPhi
    r12 =  sinPhi*cosTheta
    r13 = -cosPsi*sinPhi + sinPsi*sinTheta*cosPhi
    r21 = -sinPsi*cosPhi + cosPsi*sinTheta*sinPhi 
    r22 =  cosPsi*cosTheta
    r23 =  sinPsi*sinPhi + cosPsi*cosTheta*cosPhi
    r31 =  cosTheta*sinPhi
    r32 = -sinTheta 
    r33 =  cosTheta*cosPhi
    R = np.asarray([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]).transpose()
    return R

#Euler Angles to Quaternion: Sequence ZXY - Yaw Pitch Roll
def EulerToQuaternion(x, y, z):
    sinTheta2 = np.sin(x/2)
    cosTheta2 = np.cos(x/2)
    sinPhi2 = np.sin(y/2)
    cosPhi2 = np.cos(y/2)
    sinPsi2 = np.sin(z/2)
    cosPsi2 = np.cos(z/2)
    q0 =  cosPsi2*cosTheta2*cosPhi2 + sinPsi2*sinTheta2*sinPhi2
    q1 =  cosPsi2*sinTheta2*cosPhi2 + sinPsi2*cosTheta2*sinPhi2 
    q2 =  cosPsi2*cosTheta2*sinPhi2 - sinPsi2*sinTheta2*cosPhi2
    q3 = -cosPsi2*sinTheta2*sinPhi2 + sinPsi2*cosTheta2*cosPhi2
    q = np.asarray([q0, q1, q2, q3]).transpose()
    norm = np.sqrt((q*q).sum(axis=1))    
    for i in range(0, len(x)):
        q[i,0] = q[i,0]/norm[i]
        q[i,1] = q[i,1]/norm[i]
        q[i,2] = q[i,2]/norm[i]
        q[i,3] = q[i,3]/norm[i]
    return q

#Quaternion to Euler Angles: Sequence ZXY - Yaw Pitch Roll
def QuaternionToEuler(q):
    q0 = q[:,0]
    q1 = q[:,1]
    q2 = q[:,2]
    q3 = q[:,3]
    norm = np.sqrt((q*q).sum(axis=1))    
    for i in range(0, len(q0)):
        q[i,0] = q[i,0]/norm[i]
        q[i,1] = q[i,1]/norm[i]
        q[i,2] = q[i,2]/norm[i]
        q[i,3] = q[i,3]/norm[i]
    roll = np.arctan2((2*q1*q3+2*q0*q2), (q0*q0-q1*q1-q2*q2+q3*q3))
    pitch = -np.arcsin((2*q2*q3-2*q0*q1))
    yaw = np.arctan2((2*q1*q2+2*q0*q3), (q0*q0-q1*q1+q2*q2-q3*q3))
    E = np.asarray([pitch, roll, yaw]).transpose()
    return E 

#Quaternion to Rotation Matrix: Sequence ZXY - Yaw Pitch Roll
def QuaternionToRotMatrix(q):
    q0 = q[:,0]
    q1 = q[:,1]
    q2 = q[:,2]
    q3 = q[:,3]
    m11 = q0*q0+q1*q1-q2*q2-q3*q3
    m12 = 2*q1*q2+2*q0*q3
    m13 = 2*q1*q3-2*q0*q2
    m21 = 2*q1*q2-2*q0*q3 
    m22 = q0*q0-q1*q1+q2*q2-q3*q3
    m23 = 2*q2*q3+2*q0*q1
    m31 = 2*q1*q3+2*q0*q2
    m32 = 2*q2*q3-2*q0*q1 
    m33 = q0*q0-q1*q1-q2*q2+q3*q3
    M = np.asarray([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]).transpose()
    return M

#Runge-Kutta 4th Order Method for Integration of Quaternion
def IntegrationOfAngularVelocity(wx, wy, wz, qt_0, dt):
    #dqdt - speed in angular velocity
    q0 = qt_0[:,0]
    q1 = qt_0[:,1]
    q2 = qt_0[:,2]
    q3 = qt_0[:,3]
    dqdt_0 = 0.5*( wz*q1-wy*q2+wx*q3)
    dqdt_1 = 0.5*(-wz*q0+wx*q2+wy*q3)
    dqdt_2 = 0.5*( wy*q0-wx*q1+wz*q3)
    dqdt_3 = 0.5*(-wx*q0-wy*q1-wz*q2)
    #Runge-Kutta Method to integrate speed in angular velocity (Change In Velocity)
    #First element of a quaternion
    k1_0 = dqdt_0
    k2_0 = dqdt_0 + 0.5*k1_0*dt
    k3_0 = dqdt_0 + 0.5*k2_0*dt
    k4_0 = dqdt_0 + k3_0*dt
    qt_1_0 = dqdt_0 + 1/6*(k1_0+2*k2_0+2*k3_0+k4_0)*dt
    #Second element of a quaternion
    k1_1 = dqdt_1
    k2_1 = dqdt_1 + 0.5*k1_1*dt
    k3_1 = dqdt_1 + 0.5*k2_1*dt
    k4_1 = dqdt_1 + k3_1*dt
    qt_1_1 = dqdt_1 + 1/6*(k1_1+2*k2_1+2*k3_1+k4_1)*dt
    #Third element of a quaternion
    k1_2 = dqdt_2
    k2_2 = dqdt_2 + 0.5*k1_2*dt
    k3_2 = dqdt_2 + 0.5*k2_2*dt
    k4_2 = dqdt_2 + k3_2*dt
    qt_1_2 = dqdt_2 + 1/6*(k1_2+2*k2_2+2*k3_2+k4_2)*dt
    #Fourth element of a quaternion
    k1_3 = dqdt_3
    k2_3 = dqdt_3 + 0.5*k1_3*dt
    k3_3 = dqdt_3 + 0.5*k2_3*dt
    k4_3 = dqdt_3 + k3_3*dt
    qt_1_3 = dqdt_3 + 1/6*(k1_3+2*k2_3+2*k3_3+k4_3)*dt
    #Combine into an array
    qt_1 = np.asarray([qt_1_0, qt_1_1, qt_1_2, qt_1_3]).transpose()
    return qt_1

#############################################################################################
#                                        Frame Conversion                                   #
#############################################################################################

#Normalising Data 
def DataNorm(x, y, z):
    magnitude = np.sqrt(x*x+y*y+z*z)
    xNorm = x/magnitude
    yNorm = y/magnitude
    zNorm = z/magnitude
    Norm = pd.DataFrame([xNorm, yNorm, zNorm]).transpose()
    Norm.columns = ['x','y','z']
    return Norm

#Calculate rotation matrix through gravity and magnetic field data (Static Approach - when the phone is static)
def GraMagMatrix(uNx, uNy, uNz, uEx, uEy, uEz, uGx, uGy, uGz):
    theta = np.arccos((uNx+uEy+uGz-1)/2)
    W1 = 1/2*np.sin(theta)*(uGy-uEz)
    W2 = 1/2*np.sin(theta)*(uNz-uGz)
    W3 = 1/2*np.sin(theta)*(uEx-uNy)
    W_tot = pd.DataFrame([W1, W2, W3]).transpose()
    W_tot.columns = ['x','y','z']
    W_tot['theta'] = theta
    return W_tot



