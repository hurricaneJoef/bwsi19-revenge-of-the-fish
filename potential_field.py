#!/usr/bin/env python

import numpy as np
import sys, math, random, copy
import rospy, copy, time
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class PotentialField:
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
        self.cartPoints = []

        #[speed, angle]
        self.finalVector = [0, 0]

    def scan_callback(self, data):
        '''Checks LIDAR data'''
        self.data = data.ranges
        self.drive_callback()

    def drive_callback(self):
        '''Publishes drive commands'''
        #make sure to publish cmd here
        self.convertPoints(self.data)
        self.calcFinalVector()
        self.cmd.drive.speed = self.finalVector[0]
        self.cmd.drive.steering_angle = self.finalVector[1]
        self.drive_pub.publish(self.cmd)
        
    def calcFinalVector(self):
        #adds forward vector to sum, then reverses and inverts all distance vectors
        #sets speed based proportional to the distance at the desired angle and the average of the lidar data
        sum=[-9,9]
        point_pos=0
        for point in self.cartPoints:
            if point_pos>len(self.cartPoints)/6 and point_pos<5*len(self.cartPoints)/6:
                k=4
                point[0]=math.pow(k/float(point[0]+0.0000001),3)
                #point[0]*=-1
                point[1]=math.pow(k/float(point[1]+0.0000001),3)
                #point[1]*=-1
                sum[0]+=point[0]
                sum[1]+=point[1]
        angle=math.atan2(sum[1],sum[0])
        angle*=180/math.pi
        self.finalVector[0]=3
        self.finalVector[1]=(angle-135)/float(135)
            
        
    def convertPoints(self, points):
        #xy(1,0) @ data_pos 0, is rightmost point
        self.cartPoints = [None for x in range(len(self.data))]
        point_pos=0
        for point in points:
            angle=point_pos*270/float(len(points)-1)
            angle*=math.pi/180
            distance=points[point_pos]
            self.cartPoints[point_pos]=[math.cos(angle)*distance, math.sin(angle)*distance]
            point_pos+=1

if __name__ == "__main__":
    rospy.init_node('potential_field')
    potential_field = PotentialField()
    rospy.spin()
