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
        self.select_bin(self.data)
        self.cmd.drive.speed = self.finalVector[0]
        self.cmd.drive.steering_angle = self.finalVector[1]
        self.drive_pub.publish(self.cmd)
        
    def select_bin(self, points):
        chop=15
        mod_data=[None for x in range(0,chop)]
        x=0
        while x < chop:
            mod_data[x]=points[x*len(points)/chop:(x+1)*len(points)/chop]
            mod_data[x]=sum(mod_data[x])/len(mod_data[x])
            x+=1
        print(mod_data)
        highest=None
        x_pos=None
        for point in mod_data:
            if highest == None:
                if mod_data.index(point)>chop/6:
                    highest=mod_data[mod_data.index(point)]
                    x_pos=mod_data.index(point)
            elif highest<mod_data[mod_data.index(point)]:
                if mod_data.index(point)>chop/6 and mod_data.index(point)<5*chop/6:
                    highest=mod_data[mod_data.index(point)]
                    x_pos=mod_data.index(point)
        print("bin: "+str(x_pos))
        angle=x_pos/float(chop)+0.5/float(chop)
        angle=angle*2-1
        print(angle)
        self.finalVector[1]=angle
        
if __name__ == "__main__":
    rospy.init_node('bang_bang')
    bang_bang = BangBang()
    rospy.spin()
