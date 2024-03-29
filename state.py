#!/usr/bin/env python
import cv2
import numpy as np
import sys, math, random, copy
import rospy, copy, time
from newZed import Zed_converter
from sensor_msgs.msg import LaserScan, Joy, Image
from ackermann_msgs.msg import AckermannDriveStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from color_segmentation import cd_color_segmentation
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped
class statematch:
    SCAN_TOPIC = "/scan"
    DRIVE_TOPIC = "/drive"
    AR_TOPIC = "/ar_pose_marker"
    BUTTON_TOPIC = "/vesc/joy"
    kpfv=1
    kpf=.1
    krearforce=100
    def __init__(self):
        print("suffering")
        self.data = None
        self.cmd = AckermannDriveStamped()
        self.camera_data = Zed_converter(False, save_image = False)
        #write your publishers and subscribers here; they should be the same as the wall follower's   
        self.laser_sub = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.scan_callback, queue_size=1)
        self.ar_sub = rospy.Subscriber(self.AR_TOPIC, AlvarMarkers, self.arCallback)
        self.sound_sub = rospy.Subscriber("state", String, self.sound)
        self.sound_pub = rospy.Publisher("state", String, queue_size=1)
        self.drive_pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
	self.button_sub= rospy.Subscriber(self.BUTTON_TOPIC, Joy, self.buttonCallback, queue_size=1)
        #cartesian points -- to be filled (tuples)
        self.state=0
        #[speed, angle]
        self.finalVector = [3, 0]
        #self.state = "9"
        self.x = 0
        self.y = 0
        self.z = 0
        self.euler = None
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.dir=1
        self.go=False
        self.targetx = 0
        self.targety = 0
        self.lastdir=0
	self.start_button_on=False
    
    def statepic(self):
        print "state:"+str(self.state)
	if(self.start_button_on):
            self.state=0
            self.go=False
        if self.state==0:
            #TOD for race pf
            #print self.greenlight(self.camera_data.cv_image)
            if self.go:
                print("zoom")
		s,a=self.select_bin(self.data.ranges)
		a=self.sef(a,self.data.ranges)
                self.drive(2,a)
            else:
                self.go=self.greenlight(self.camera_data.cv_image)
                self.drive(0,0)
                print self.go
        elif self.state==1:
            if min(self.data.ranges[160:200])>1.5:
                self.drive(5,self.wf(1,1))#TODO turnpike between lines full speed
            else:
                s,a=self.select_bin(self.data.ranges)
                self.drive(2,a)
        elif self.state==2:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TOD end turnpike left wall
        elif self.state==3:
            if max(self.data.ranges)>5:
                s,a=self.select_bin(self.data.ranges)
                self.drive(s,a)
            else:
		self.drive(1.5,self.wf(-1,0.6))#TODO left wall follower
            
        elif self.state==4:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO beaver baller pf/ car wash
        elif self.state==5:
            s,a=self.select_bin(self.data.ranges)
            a=self.sef(a,self.data.ranges)
            self.drive(1.5,a)#TODO graveyard pf
        elif self.state==6:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO python path  lwf
        elif self.state==7:
            if min(self.data.ranges[160:200])>1.5:
                self.drive(5,self.wf(1,1))#TODO turnpike between lines full speed
            else:
                s,a=self.select_bin(self.data.ranges)
                self.drive(2,a)#TODO turnpike between lines full speed
		#TODO other way turnpike full speed
        elif self.state==8:
            self.drive(1.5,self.wf(1,0.5))#TODO  end of turnpike right wall /bob's brick bypass
        elif self.state==9:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO bridge pf
        elif self.state==10:
            None#TODO 10 singdirthen wall follower
            dirnow=0#self.signdir(self.camera_data.cv_image)
            print("dirnow:"+str(dirnow))
            if dirnow != 0:
                self.dir=dirnow
            print("sign dir:"+str(self.dir))
            self.drive(1,self.wf(self.dir,0.5))
        elif self.state==11:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO rwf/pf
        elif self.state==12:
            self.drive(1,self.wf(-1,0.6))#TODO pf
        elif self.state==13:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO pf
        elif self.state==14:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO pf
        elif self.state==15:
            s,a=self.select_bin(self.data.ranges)
            self.drive(s,a)#TODO rwf
        elif self.state==16:
            self.drive(1.5,self.wf(1,1))#TODO rwf
        elif self.state==17:
            self.drive(2,self.wf(1,1))#TODO 17 full speed then pull over
        else:
            self.state=0
            self.go=False
            print("yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet yeet ")
    def sound(self, state):
        if state != None:
            a = 0
            
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
  #      if self.state != "1" and self.state != "2" and self.state != "5" and self.cmd.drive.speed == 0
		#if tags.markers[i].id == 1:
                #self.state = "1"
            #elif tags.markers[i].id == 2:
             #   self.state = "2"
##            elif tags.markers[i].id == 5:
#/                self.state = "5"
	if len(tags.markers)>0:
		if tags.markers[0].id>0:
		    if tags.markers[0].id<20:
                        self.state = tags.markers[0].id
        	pass
    def buttonCallback(self, data):
	if data.buttons[1]==1:
		self.start_button_on= True
	else:
		self.start_button_on= False
    
    def scan_callback(self, data):
        '''Checks LIDAR data'''
        self.data = data
        self.drive_callback()

    def drive_callback(self):
        '''Publishes drive commands'''
        #make sure to publish cmd here
        self.statepic()
        self.cmd.drive.speed = self.finalVector[0]
        self.cmd.drive.steering_angle = self.finalVector[1]
        self.drive_pub.publish(self.cmd)
        
    def quatToAng3D(self, quat):
        euler = euler_from_quaternion((quat.x,quat.y,quat.z,quat.w))
        return euler
    
    def wf(self,dir,dist):
        if dir == 1:
            return self.right_wall_follow(dist)
        elif dir == -1:
            return self.left_wall_follow(dist)
        else:
            return 0
    def right_wall_follow(self,distin):
        # if lidar data has not been received, do nothing
        if self.data == None:
            print "No data"
            return 0
        dist = self.right()
	print(dist)
        disi = self.righi()
        if dist > distin-0.3:
            if dist > distin and disi < 180:
                angle = -0.3
            if dist > distin and disi < 240:
                angle = -0.05
            elif dist > distin and disi > 240:
                angle = 0.05
            elif dist > distin:
                angle = 0
            else:
                ang = disi-180
                angle = ang/1000.0
        else:
           angle = 0.3
        print "angle", angle
        #make sure to publish cmd here
	return angle
    def right(self):
        smallest = 1000
        index = 0
        for i in range(50,300):
            if self.data.ranges[i] < smallest:
                smallest = self.data.ranges[i]
                index = i
        return smallest
    def righi(self):
        smallest = 1000
        index = 0
        for i in range(50,300):
            if self.data.ranges[i] < smallest:
                smallest = self.data.ranges[i]
                index = i
        return index
    def left_wall_follow(self,distin):
        # if lidar data has not been received, do nothing
        if self.data == None:
            print "No data"
            return 0
        self.cmd.drive.speed = 1
        dist = self.left()
        disi = self.lefi()
        if dist > distin-0.3:
            if dist > distin and disi > 900:
                angle = 0.3
            elif dist > distin and disi > 760:
                angle = 0.05
            elif dist > distin and disi < 760:
                angle = -0.05
            elif dist > distin:
                angle = 0
            else:
                ang = 84-disi
                angle = ang/1000.0
        else: 
            angle = -0.3
        print "angle", angle
        #make sure to publish cmd here
	return angle
    def left(self):
        smallest = 1000
        index = 0
        for i in range(250):
            if self.data.ranges[i+780] < smallest:
                smallest = self.data.ranges[i+780]
                index = i+780
        return smallest
    def lefi(self):
        smallest = 1000
        index = 0
        for i in range(250):
            if self.data.ranges[i+780] < smallest:
                smallest = self.data.ranges[i+780]
                index = i+780
        return index
    def sef(self,angle,points):
        if min(points[50:450])<.5 or min(points[630:1030])<.5:
            angle=0
            if min(points[50:450])<.4:
                angle=-0.3
            if min(points[630:1030])<.4:
                angle=0.3
        return angle
    def signdir(self,img,threshold=.6,bestmatch=False):
       img_rgb = img
       img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
       bw = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)[1]
#        template = cv2.imread("images/LEFT.png",0)
#        lres = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
#        template = cv2.imread("images/RIGHT.png",0)
#        rres = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
       template = cv2.imread("LEFTS.png",0)
       lress = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
       template = cv2.imread("RIGHTS.png",0)
       rress = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
#        template = cv2.imread("images/LEFTxS.png",0)
#        lresxs = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
#        template = cv2.imread("images/RIGHTxS.png",0)
#        rresxs = cv2.matchTemplate(bw,template,cv2.TM_CCOEFF_NORMED)
       output=0
       if(bestmatch):
           output = 0
           if(np.max(lres)<np.max(rres)):
               output=1
           else:
               output=-1
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
       if self.lastdir==output:
           return output
       self.lastdir=output
       return(0)
    import cv2
    #import imutils
    import numpy as np
    import pdb

    def cd_color_segmentation(img,colorval=[[ 52, 66, 61],[ 98, 255, 148]], show_image=False):
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
        low_range  = np.array(colorval[0])
        high_range = np.array(colorval[1]) #120,255,255
    
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
        box=cd_color_segmentation(img,colorval=[[63, 119, 75],[98, 255, 100]])#[21, 129, 63],[83, 255, 205]  [ 17, 175, 23],[ 82, 255, 148]
        if box[1][1]<300:
            if self.size_calc(box)>200:
                return True
        return False
    def antigreenlight(self,img):
        box=cd_color_segmentation(img,colorval=[[88, 135, 179],[144, 255, 255]])#[21, 129, 63],[83, 255, 205]  [ 17, 175, 23],[ 82, 255, 148]
        if box[1][1]<300:
            if self.size_calc(box)>1000:
                return True
        return False
    def size_calc(self,box):
        """ calculate the x and y size of the box in pixels"""
        pix_width  = box[1][0] - box[0][0]
        pix_height = box[1][1] - box[0][1]    
        return pix_width*pix_height   
    def pol2cart(self,r, theta):
       	x = r * np.cos(theta)
       	y = r * np.sin(theta)
       	return(x, y)
    def cart2pol(self,x,y):
       	rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    def pf(self,data):
        force=[]
        for i in range(len(data.ranges)):
            x,y=self.pol2cart(self.kpf/(data.ranges[i]**3),  i*data.angle_increment-np.pi/4)
            force.append([x,y])
        force.append([0,self.krearforce])
        total=np.sum(force, axis=0)
        speed,angle=self.cart2pol(total[0],total[1])
        angle-=np.pi/2
        speed*=self.kpfv
        return speed , angle
    def select_bin(self,points):
        chop=15
        mod_data=[None for x in range(0,chop)]
        x=0
        while x < chop:
            mod_data[x]=points[x*len(points)/chop:(x+1)*len(points)/chop]
            mod_data[x]=sum(mod_data[x])/len(mod_data[x])
            x+=1
        #print(mod_data)
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
        #print("bin: "+str(x_pos))
        angle=x_pos/float(chop)+0.5/float(chop)
        angle=angle*2-1
        angle*=0.7
        #print(angle)
        return 1.5,angle  
        
        
        
        
        
        
        
    def drive(self,speed,angle):
        self.finalVector=[speed,angle]
	print(self.finalVector)
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
    rospy.init_node('state_machine')
    state_drive = statematch()
    rospy.spin()
