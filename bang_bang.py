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
        self.finalVector = [3, 0]

    def scan_callback(self, data):
        '''Checks LIDAR data'''
        self.data = data.ranges
        self.drive_callback()

    def drive_callback(self):
        '''Publishes drive commands'''
        #make sure to publish cmd here
        self.bang_bang_go(self.data)
        self.cmd.drive.speed = self.finalVector[0]
        self.cmd.drive.steering_angle = self.finalVector[1]
        self.drive_pub.publish(self.cmd)
        
    def bang_bang_go(self, points):
        desired_dis=3
        point_far=points[len(points)/3]
        point_right=points[len(points)/6]
        turn=0
        if point_right<desired_dis:
            turn+=0.5
        if point_right>desired_dis:
            turn-=0.5
        if point_right>point_far:
            turn+=0.5
        if point_right<point_far:
            turn-=0.5
        self.finalVector[1]=turn
if __name__ == "__main__":
    rospy.init_node('bang_bang')
    bang_bang = BangBang()
    rospy.spin()
