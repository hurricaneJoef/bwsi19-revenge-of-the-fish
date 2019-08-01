#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
class Safety:
    DRIVE_IN_TOPIC = '/drive'
    DRIVE_OUT_TOPIC = '/vesc/high_level/ackermann_cmd_mux/input/default'
    SCAN_TOPIC = '/scan'

    def __init__(self):
        self.sub_scan = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, callback=self.scan_callback)
        self.sub_drive = rospy.Subscriber(self.DRIVE_IN_TOPIC, AckermannDriveStamped, callback=self.drive_callback)
        self.pub_drive = rospy.Publisher(self.DRIVE_OUT_TOPIC, AckermannDriveStamped, queue_size=1)
	#print(self.scan.angle_min)

    def scan_callback(self, msg):
        self.scan = msg

    def drive_callback(self, msg):
        if self.is_safe(msg):
            self.pub_drive.publish(msg)

    def is_safe(self, msg):
        closest=None
        base=1.5
        dist=base
        for data in range(len(self.scan.ranges)/6,5*len(self.scan/ranges)/6):
            if closest==None and self.scan.ranges[data]<dist:
                closest==data
                dist=self.scan.ranges[data]
        if not closest==None:
            if(base-dist<0.5)
                msg.drive.speed*=0.33
            else:
                msg.drive.speed*=dist/base
            obj_angle=(closest-len(self.scan.ranges)/2)/len(self.scan.ranges)/2
            dr
                
rospy.init_node('safety')
safety = Safety()
rospy.spin()
