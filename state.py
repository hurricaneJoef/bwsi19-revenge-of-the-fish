#!/usr/bin/env python
import cv2
import numpy as np
import sys, math, random, copy
import rospy, copy, time
from newZed import Zed_converter
from sensor_msgs.msg import LaserScan, Joy, Image
from ackermann_msgs.msg import AckermannDriveStamped
from color_segmentation import cd_color_segmentation
from cv_bridge import CvBridge, CvBridgeError
class statematch:
    SCAN_TOPIC = "/scan"
    DRIVE_TOPIC = "/drive"

    def __init__(self):
        print("suffering")
        self.data = None
        self.cmd = AckermannDriveStamped()
        self.camera_data = Zed_converter(False, save_image = False)
        #write your publishers and subscribers here; they should be the same as the wall follower's   
        self.laser_sub = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.scan_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        #cartesian points -- to be filled (tuples)

        #[speed, angle]
        self.finalVector = [3, 0]
    
    def statepic(self):
        #TODO save ar tag val
        print self.state
        if self.state==0:
            #TODO for race pf
            
            if self.go:
            
            else:
                self.go=greenlight(self.camera_data.cv_image)
		print self.go
        elif self.state==1:
            #TODO turnpike between lines full speed
        elif self.state==2:
            #TODO end turnpike left wall
        elif self.state==3:
            #TODO left wall follower
            
        elif self.state==4:
            #TODO beaver baller pf/ car wash
        elif self.state==5:
            #TODO graveyard pf
        elif self.state==6:
            #TODO python path  pf
        elif self.state==7:
            #TODO other way turnpike full speed
        elif self.state==8:
            #TODO  end of turnpike right wall /bob's brick bypass
        elif self.state==9:
            #TODO bridge pf
        elif self.state==10:
            #TODO 10 singdirthen wall follower
        elif self.state==11:
            #TODO rwf/pf
        elif self.state==12:
            #TODO rwf
        elif self.state==13
            #TODO pf
        elif self.state==14
            #TODO pf
        elif self.state==15
            #TODO rwf
        elif self.state==16
            #TODO rwf
        elif self.state==17
            #TODO 17 full speed then pull over
        else:
            self.state=0
    
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
    def signdir(self,img,threshold=.6,bestmatch=False):
       img_rgb = img
       img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
       bw = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)[1]
#        template = cv2.imread("images/LEFT.png",0)
#        lres = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
#        template = cv2.imread("images/RIGHT.png",0)
#        rres = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
       template = cv2.imread("images/LEFTS.png",0)
       lress = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
       template = cv2.imread("images/RIGHTS.png",0)
       rress = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
#        template = cv2.imread("images/LEFTxS.png",0)
#        lresxs = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
#        template = cv2.imread("images/RIGHTxS.png",0)
#        rresxs = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
       output=0
       if(bestmatch):
           output = 0
#            if(np.max(lres)<np.max(rres)):
#                output=1
#            else:
#                output=-1
       else:
#            if(np.max(lres) >= threshold) or (np.max(lress)>= threshold) or (np.max(lresxs)>= threshold):
#                output-=1
#            if(np.max(rres) >= threshold) or (np.max(rress)>= threshold) or (np.max(rresxs)>= threshold):
#                output+=1
           if np.max(lress)>= threshold:
               output-=1
           if np.max(rress)>= threshold:
               output+=1
       rospy.loginfo("output: {}".format(output))
       return output
   import cv2
    #import imutils
    import numpy as np
    import pdb

    def cd_color_segmentation(img,colorval=[[ 31, 154, 16],[ 61, 255, 217]], show_image=False):
        """
	    Implement the cone detection using color segmentation algorithm
	        Input:
	        img: np.3darray; the input image with a cone to be detected
    	    Return:
	        bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
	    	    (x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
        """
        # convert from rgb to hsv color space (it might be BGR)
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # define lower and upper bound of image values
        # TO DO!
        low_range  = colorval[0]
        high_range = colorval[1] #120,255,255
    
        # create mask for image with overlapping values
        mask = cv2.inRange(new_img, low_range, high_range)
    
        # filter the image with bitwise and
        filtered = cv2.bitwise_and(new_img, new_img, mask=mask)
    
        # find the contours in the image
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        x1, y1, x2, y2 = 0, 0, 0, 0
        if len(contours) != 0:
    	# find contour with max area, which is most likely the cone
            # Solution note: max uses an anonymous function in this case, we can also use a loop...
            contours_max = max(contours, key = cv2.contourArea)
    
    	# Find bounding box coordinates
            x1, y1, x2, y2 = cv2.boundingRect(contours_max)
    
    	# Draw the bounding rectangle
            cv2.rectangle(img, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)
    
        if show_image:
            cv2.imshow("Color segmentation", img)
            key = cv2.waitKey()
            if key == 'q':
                cv2.destroyAllWindows()
    
        # Return bounding box
        #return img
        return ((x1, y1), (x1 + x2, y1 + y2))    
    def size_calc(self,data):
		""" calculate the x and y size of the box in pixels"""
		size=[data[0][0]-data[1][0],data[1][1]-data[0][1]]
		return size       
    def greenlight(self,img):
        box=cd_color_segmentation(img,colorval=[[ 31, 154, 16],[ 61, 255, 217]])
        if box[1][1]>150:
            if self.size_calc(box)>400:
                return true
        return false
    def size_calc(self,box):
        """ calculate the x and y size of the box in pixels"""
        pix_width  = box[1][0] - box[0][0]
        pix_height = box[1][1] - box[0][1]    
        return pix_width*pix_height   
       
       
       
       
       
       
       
       
       
       
           
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
    rospy.init_node('state machine')
    state_drive = statematch()
    rospy.spin()
