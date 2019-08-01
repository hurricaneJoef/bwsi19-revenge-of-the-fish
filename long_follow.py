#!/usr/bin/env python

import numpy as np
import sys, math, random, copy
import rospy, copy, time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class BangBang:
    SCAN_TOPIC = "/scan"
    DRIVE_TOPIC = "/drive"

    def __init__(self):
        print("suffering")
        self.data = None
        self.cmd = AckermannDriveStamped()

        #write your publishers and subscribers here; they should be the same as the wall follower's   
        self.laser_sub = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.scan_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        #cartesian points -- to be filled (tuples)

        #[speed, angle]
        self.finalVector = [1.5, 0]

    def scan_callback(self, data):
        '''Checks LIDAR data'''
        self.data = data.ranges
        self.drive_callback()

    def drive_callback(self):
        '''Publishes drive commands'''
        #make sure to publish cmd here
        self.select_point(self.data)
        self.cmd.drive.speed = self.finalVector[0]
        self.cmd.drive.steering_angle = self.finalVector[1]
        self.drive_pub.publish(self.cmd)
        
    def select_point(self, points):
        low_bound=2
        pos=0
        for point in points:
            if point<low_bound:
                points[pos]=0
            pos+=1
        pos=0
        bins=[]
        while pos<len(points):
            if points[pos]=0 and not points.index(0,pos+1)==pos+1:
                splice=points[pos:points.index(0,pos+1)]
                bins.append(sum(splice)/float(len(splice)),sum(range(pos,points.index(0,pos+1)))/len(splice))
            else:
                pos+=1
        angle=0
        highest=0
        for data in bins:
            if data[0]>highest:
                highest=data[0]
                angle=data[1]
        self.finalVector[1]=angle/float(len(points))*2-1
        
if __name__ == "__main__":
    rospy.init_node('bang_bang')
    bang_bang = BangBang()
    rospy.spin()
