#!/usr/bin/env python

import numpy as np
import sys, math, random, copy
import rospy, copy, time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

def func_pf(scan):
    #cartesian points -- to be filled (tuples)
    cartPoints = convertPoints(scan)
    #[speed, angle]
    finalVector = calcFinalVector(scan, cartPoints)
    return finalVector
    
        
def calcFinalVector(scan, data):
    #adds forward vector to sum, then reverses and inverts all distance vectors
    #sets speed based proportional to the distance at the desired angle and the average of the lidar data
    sum=[-2.12,2.12]
    for point in data:
        point[0]=1/float(point[0])
        point[0]*=-1
        point[1]=1/float(point[1])
        point[1]*=-1
        sum[0]+=point[0]
        sum[1]+=point[1]
    angle=math.atan2(sum[1],sum[0])
    angle*=180/math.pi
    dis_at_angle=scan[round(len(data)*angle/float(270))]
    average=0
    for point in scan
        average+=point
    average/=len(scan)
    finalVector=[0,0]
    finalVector[0]=dis_at_angle/average
    finalVector[1]=(angle-135)/float(135)
    return finalVector
            
        
def convertPoints(points):
    #xy(1,0) @ data_pos 0, is rightmost point
    cartPoints = [None for range(0,len(self.data)]
    point_pos=0
    for point in points:
        angle=point_pos*270/float(len(points-1))
        angle*=math.pi/180
        distance=points[point_pos]
        cartPoints[point_pos]=[math.cos(angle)*distance, math.sine(angle)*distance]
        point_pos+=1
    return cartPoints
