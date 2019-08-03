#/usr/bin/env python

import rospy
import time
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan, Joy, Image
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped
import math


DRIVE_TOPIC = "/drive"
SCAN_TOPIC = "/scan"
AR_TOPIC = "/ar_pose_marker"

class ARDrive(object):
    def __init__(self):
        rospy.init_node("ar")
        #initialize publishers and subscribers
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size = 1)
        self.scan_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self.driveCallback)
        self.map_scan_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.findPos)
        self.ar_sub = rospy.Subscriber(AR_TOPIC, AlvarMarkers, self.arCallback)
        self.sound_sub = rospy.Subscriber("state", String, self.sound)
        self.sound_pub = rospy.Publisher("state", String, queue_size=1)
        
        #initialize cmd object
        self.cmd = AckermannDriveStamped()
        self.cmd.drive.speed = 0
        self.cmd.drive.steering_angle = 0

        self.state = "9"
        self.x = 0
        self.y = 0
        self.z = 0
        self.euler = None
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.targetx = 0
        self.targety = 0
    
    def sound(self, state):
        if state != None:
            a = 0
        

    def driveCallback(self, data):
        '''LIDAR callback, sets drive commands'''
        self.setTarget()
        x = self.targetx-self.x
        y = self.targety-self.y
        
        #print data.ranges[540]
        #print self.state
        if self.state == ("9") and data.ranges[540] < 0.4:
            self.cmd.drive.speed = 0
            #print "yes"
        elif self.state == ("1") and data.ranges[540] > 1.5:
            self.state = "3"
        elif self.state == ("2") and data.ranges[540] > 1.5:
            self.state = "4"
        elif self.state == ("5") and data.ranges[540] > 1.5:
            self.state = "7"
        elif self.state == ("3") and data.ranges[540] < 0.4:
            self.state = "a"
        elif self.state == ("4") and data.ranges[540] < 0.4:
            self.state = "b"
        elif self.state == ("7") and data.ranges[540] < 0.4:
            self.state = "c"
        elif self.state == ("a") and data.ranges[540] > 2.5:
            self.state = "9"
        elif self.state == ("b") and data.ranges[540] > 2.5:
            self.state = "9"
        elif self.state == ("c") and data.ranges[540] > 2.5:
            self.state = "9"
        #TODO: Set drive commands according to the current state

    def findPos(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z

        self.euler = self.quatToAng3D(data.pose.pose.orientation)
        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]

    def arCallback(self, tags):
        '''Callback when an AR tag is detected'''
        #TODO: Write your state changes here
        if self.state != "1" and self.state != "2" and self.state != "5" and self.cmd.drive.speed == 0:
            if tags.markers[i].id == 1:
                self.state = "1"
            elif tags.markers[i].id == 2:
                self.state = "2"
            elif tags.markers[i].id == 5:
                self.state = "5"
        print tags
        pass
    
    def setTarget(self):
        if self.state == ("9"):
            #print "yey"
            self.cmd.drive.speed = 0.5
            self.cmd.drive.steering_angle = -0.03
        elif self.state == ("1"):
            self.cmd.drive.speed = -0.5
            self.cmd.drive.steering_angle = -0.03
        elif self.state == ("2"):
            self.cmd.drive.speed = -0.5
            self.cmd.drive.steering_angle = -0.03
        elif self.state == ("5"):
            self.cmd.drive.speed = -0.5
            self.cmd.drive.steering_angle = -0.03
        elif self.state == ("3"):
            self.cmd.drive.speed = 0.5
            self.cmd.drive.steering_angle = 1
        elif self.state == ("4"):
            self.cmd.drive.speed = 0.5
            self.cmd.drive.steering_angle = 0.05
        elif self.state == ("7"):
            self.cmd.drive.speed = 0.5
            self.cmd.drive.steering_angle = -0.05
        elif self.state == ("a"):
            self.cmd.drive.speed = -0.5
            self.cmd.drive.steering_angle = 1
        elif self.state == ("b"):
            self.cmd.drive.speed = -0.5
            self.cmd.drive.steering_angle = 0.05
        elif self.state == ("c"):
            self.cmd.drive.speed = -0.5
            self.cmd.drive.steering_angle = -0.05
        


    def quatToAng3D(self, quat):
        euler = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        return euler

def main():
    try:
        ic = ARDrive()
        rospy.Rate(100)
        while not rospy.is_shutdown():
            ic.drive_pub.publish(ic.cmd)
            #print ic.cmd.drive.speed
            ic.sound_pub.publish(ic.state)
    except rospy.ROSInterruptException:
        exit()

if __name__ == "__main__":
    main()



    

