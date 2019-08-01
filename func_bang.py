#!/usr/bin/env python

import numpy as np
import sys, math, random, copy
import rospy, copy, time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

def bang_control(scan, speed=3):
    #returns drive speed and angle as a list 
    finalVector = [speed, 0]
    desired_dis=3
    point_far=scan[len(scan)/3]
    point_right=scan[len(scan)/6]
    turn=0
    if point_right<desired_dis:
        turn+=0.5
    if point_right>desired_dis:
        turn-=0.5
    if point_right>point_far:
        turn+=0.5
    if point_right<point_far:
        turn-=0.5
    finalVector[1]=turn
    return finalVector

