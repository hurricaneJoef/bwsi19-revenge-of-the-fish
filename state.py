n

import rospy
import time
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan, Joy, Image
from ar_track_alvar_msgs.msg import AlvarMarkers
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped
import sys, math
import cv2
from newZed import Zed_converter

import cv2
import numpy as np
import sys, math, random, copy
import rospy, copy, time
from newZed import Zed_converter
from sensor_msgs.msg import LaserScan, Joy, Image
from ackermann_msgs.msg import AckermannDriveStamped
from ar_track_alvar_msgs.msg import AlvarMarkers
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped


DRIVE_TOPIC = "/drive"
SCAN_TOPIC = "/scan"
AR_TOPIC = "/ar_pose_marker"
BUTTON_TOPIC = "/vesc/joy"

class ARDrive(object):
    def __init__(self):
        rospy.init_node("ar")
        #initialize publishers and subscribers
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size = 1)
        self.scan_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self.driveCallback)
        self.ar_sub = rospy.Subscriber(AR_TOPIC, AlvarMarkers, self.arCallback)
        self.button_sub= rospy.Subscriber(self.BUTTON_TOPIC, Joy, self.buttonCallback, queue_size=1)
        
        #initialize cmd object
        self.cmd = AckermannDriveStamped()
        self.cmd.drive.speed = 0
        self.cmd.drive.steering_angle = 0
        self.targetx = 0
        self.targety = 0
        self.x = 0
        self.y = 0

        self.camera_data = Zed_converter(False, save_image = False)

        self.state = -1
        self.x = 0
        self.y = 0
        self.z = 0
        self.euler = None
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.sd = 0

        self.start_button_on=False

    def buttonCallback(self, data):
	    if data.buttons[1] == 1:
		    self.start_button_on = True
	    if data.button[2] == 1:
            self.start_button_on = False
            self.state = -1
        
        

    def driveCallback(self, data):
        '''LIDAR callback, sets drive commands'''
        self.setTarget()
        x = self.targetx-self.x
        y = self.targety-self.y

        if self.state == -1:
            self.cmd.drive.speed = 0
            if self.start_button_on and self.greenlight(self.camera_data.cv_image):
                self.state = 0
        elif self.state == 0:
            self.select_bin(self.data.ranges, 45)
            if self.right() < 0.2 and self.righi() > 180:
                self.cmd.drive.steering_angle = (self.righi()-180)/600.0
            if self.left() < 0.2 and self.lefi() < 900:
                self.cmd.drive.steering_angle = -(900-self.lefi())/600.0
        elif self.state == 1:
            self.right_wall_follow(0.4)
        elif self.state == 2:
            self.right_wall_follow(0.4)
        elif self.state == 3:
            self.select_bin(self.data.ranges, 15)
        elif self.state == 4:
            self.select_bin(self.data.ranges, 15)
        elif self.state == 5:
            self.select_bin(self.data.ranges, 15)
            self.grave()
        elif self.state == 6:
            self.select_bin(self.data.ranges, 15)
        elif self.state == 7:
            self.right_wall_follow(0.4)
        elif self.state == 8:
            self.right_wall_follow(0.4)
        elif self.state == 9:
            self.select_bin(self.data.ranges, 15)
        elif self.state == 10:
            self.right_wall_follow_2(0.4)
        elif self.state == 11:
            self.right_wall_follow(0.4)
        elif self.state == 12:
            self.right_wall_follow(0.8)
        elif self.state == 13:
            self.left_wall_follow(0.4)
        elif self.state == 14:
            #self.redbox(self.camera_data.cv_image)
            self.left_wall_follow(0.4)
        elif self.state == 15:
            self.right_wall_follow(0.6)
        elif self.state == 16:
            self.right_wall_follow(0.6)
        elif self.state == 17:
            self.right_wall_follow(0.6)
        
        self.cmd.drive.speed = 2

        print self.state

        #make sure to publish cmd here
        self.drive_pub.publish(self.cmd)

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
        if tags.markers[i].id > 0 and tags.markers[i].id < 21:
            self.state = tags.markers[i].id
        pass


    
    
    def right_wall_follow(self, distwant):
        # if lidar data has not been received, do nothing
        if self.data == None:
            print "No data"
            return 0

        dist = self.right()
        disi = self.righi()
        if dist > distwant/2.0:
            if dist > distwant and disi < 180:
                self.cmd.drive.steering_angle = -0.3
            elif dist > distwant and disi < 240:
                self.cmd.drive.steering_angle = -0.05
            elif dist > distwant and disi > 240:
                self.cmd.drive.steering_angle = 0.3
            elif dist > distwant:
                self.cmd.drive.steering_angle = 0
            else:
                ang = disi-180
                self.cmd.drive.steering_angle = ang/600.0
        else:
            self.cmd.drive.steering_angle = 0.3
    
    def right(self):
        smallest = 1000
        index = 0
        for i in range(300):
            if self.data.ranges[i] < smallest:
                smallest = self.data.ranges[i]
                index = i
        return smallest

    def righi(self):
        smallest = 1000
        index = 0
        for i in range(300):
            if self.data.ranges[i] < smallest:
                smallest = self.data.ranges[i]
                index = i
        return index

    


    
    
    def left_wall_follow(self, distwant):
        # if lidar data has not been received, do nothing
        if self.data == None:
            print "No data"
            return 0

        dist = self.left()
        disi = self.lefi()
        if dist > distwant/2.0:
            if dist > distwant and disi > 900:
                self.cmd.drive.steering_angle = 0.3
            elif dist > distwant and disi > 760:
                self.cmd.drive.steering_angle = 0.05
            elif dist > distwant and disi < 760:
                self.cmd.drive.steering_angle = -0.3
            elif dist > distwant:
                self.cmd.drive.steering_angle = 0
            else:
                ang = 900-disi
                self.cmd.drive.steering_angle = -ang/600.0
        else:
            self.cmd.drive.steering_angle = -0.3
    
    def left(self):
        smallest = 1000
        index = 0
        for i in range(300):
            if self.data.ranges[i+780] < smallest:
                smallest = self.data.ranges[i+780]
                index = i+780
        return smallest

    def lefi(self):
        smallest = 1000
        index = 0
        for i in range(300):
            if self.data.ranges[i+780] < smallest:
                smallest = self.data.ranges[i+780]
                index = i+780
        return index
    
    def setTarget():
        self.targetx = self.data.ranges[24] + self.data.ranges[1056]
        self.targety = self.data.ranges[484] + self.data.ranges[596]
    
    def select_bin(points, chop):
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
        if min(self.data.ranges[0:360])<.3 or min(self.data.ranges[720:1080])<.3:
            angle=0

        #print(angle)
        #return [1.5,angle]
        self.cmd.drive.steering_angle = angle

    def grave(self):
        smallest = 1000
        index = 0
        for i in range(140):
            if self.data.ranges[i+400] < smallest:
                smallest = self.data.ranges[i]
                index = i
        if smallest < 0.2:
            self.cmd.drive.steering_angle = 0.3
        smallest = 1000
        index = 0
        for i in range(140):
            if self.data.ranges[i+540] < smallest:
                smallest = self.data.ranges[i+780]
                index = i+780
        if smallest < 0.2:
            self.cmd.drive.steering_angle = -0.3
    
    def setdefx(self):
        self.x = self.cmd.drive.steering_angle

    def setdefy(self):
        self.y = self.cmd.drive.speed

    def sift_det(self, source):

        rects = None

        # TODO:
        # Sets up match count between images for sensitivity of detection - choose your value!
        MIN_MATCH_COUNT = 10

        # If VideoCapture(feed) doesn't work, manually try -1, 0, 1, 2, 3 (if none of those work, 
        # the webcam's not supported!)
        #cam = cv2.VideoCapture(args["source"])

        # Reads in the image
        #img1 = cv2.imread(image, 0)                      

        # Labels the image as the name passed in    
        #if args["label"] is not None:
        #    label = args["label"]
        #else:
        #    # Takes the name of the image as the name
        #    if image[:2] == "./":
        #        label = label = (image.split("/"))[2]
        #    else:
        #        label = image[2:-4]
        label = "Oneway"

        ################################################### Set up Feature Detection

        # Create a the SIFT Detector Object
        try:
            orb = cv2.ORB_create()
        except AttributeError:
            print("Install 'opencv-contrib-python' for access to the xfeatures2d module")

        # Compute keypoints
        #kp, des = orb.detectAndCompute(img1, None)

        FLANN_INDEX_KDTREE = 0
        # Option of changing 'trees' and 'checks' values for different levels of accuracy
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    
        search_params = dict(checks = 50)                                 

        # Fast Library for Approximate Nearest Neighbor Algorithm
        # Creates FLANN object for use below
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        frame = source

        ################################################### Shape Computation

        # TODO:
        # What color space does OpenCV read images in, and what color space do we want process?
        # Check out cvtColor! <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html>
        # Read in the image from the camera 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # TODO: 
        # Set up your HSV threshold bounds - [Hue, Saturation, Value]
        lower = np.array([0, 0, 200], dtype = "uint8")
        upper = np.array([255, 255, 255], dtype = "uint8")     

        # TODO: 
        # Check inRange() <https://docs.opencv.org/3.0-beta/modules/core/doc/operations_on_arrays.html?highlight=inrange#invert>
        # Create mask for image with overlapping values
        mask = cv2.inRange(img, lower, upper)

        # TODO:
        # What parameters work best for thresholding? <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold>
        imgThresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 3, 1)
        
        # TODO:
        # This is OpenCV's call to find all of the contours
        # Experiment with different algorithms (cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 
        # in the parameters of cv2.findContours!
        # <https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findContours>
        # The underscores represent Python's method of unpacking return values, but not using them
        _, contours, _ = cv2.findContours(imgThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Optional TODO:
        # Optional processing of contours - do we want to remove all non-rectangle shapes from contours?
        # Read the documentation on approxPolyDP <https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html>

        # TODO:
        # Orders contours - but by what metric? Check the "key" options <https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html>
        # Ex. key = cv2.contourVectorLen() (Would order the contours by vector length - not an actual function, but this is how you would set the "key")
        # Python's "sorted" function applies a "key" set lambda function to each element within an array, this is not a traditional dictionary "key"
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[1:]   # Removes contouring of display window

        if len(contours) != 0:
            # TODO:  
            # Draws the max of the contours arrays with respect to the "key" chosen above
            contours_max = max(contours, key = cv2.contourArea)

            # Find bounding box coordinates
            rect = cv2.boundingRect(contours_max)
            x, y, w, h = rect

            # TODO:
            # Calculates area of detection - what detection area should be the lower bound?
            if w*h > 20:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                rects = ((x, y), (x+w, y+h))
        
        g = frame

        if(rects != None and self.size_calc(rects) > 20000):
                    left = rects[0][0] + 30
                    top = rects[0][1] + 30
                    right = rects[1][0] -  30
                    bottom = rects[1][1]

                    c3 = right - left
                    c1 = abs(right - left)
                    c2 = abs(top - bottom)/5

                    nx = (left+right)/2
                    ny = (top+bottom)/2

                    sum1 = 0
                    for i in range(c1/2+left):
                        for j in range(c2+top):
                            sum1 += g[i][j][0]+g[i][j][1]+g[i][j][2]

                    sum2 = 0
                    for i in range(c1/2+left):
                        for j in range(c2+top):
                            sum2 += g[i+c1/2][j][0]+g[i+c1/2][j][1]+g[i+c1/2][j][2]

                    #print sum1 - sum2
                    if sum1 - sum2 > 0:
                        self.sd = 1
                        print "left"
                    else:
                        self.sd = -1
                        print "right"
    
    def left_wall_follow_2(self, distwant):
        # if lidar data has not been received, do nothing
        if self.data == None:
            print "No data"
            return 0

        dist = self.left()
        disi = self.lefi()

        if dist > distwant/2.0:
            if dist > distwant and disi > 760:
                self.cmd.drive.steering_angle = 0.05
            elif dist > distwant and disi < 760:
                self.cmd.drive.steering_angle = -0.05
            elif dist > distwant:
                self.cmd.drive.steering_angle = 0
            else:
                ang = 900-disi
                self.cmd.drive.steering_angle = ang/600.0
        else:
            self.cmd.drive.steering_angle = -0.3
    
    def right_wall_follow_2(self, distwant):
        # if lidar data has not been received, do nothing
        if self.data == None:
            print "No data"
            return 0

        dist = self.right()
        disi = self.righi()

        if dist > distwant/2.0:
            if dist > distwant and disi < 240:
                self.cmd.drive.steering_angle = -0.05
            elif dist > distwant and disi > 240:
                self.cmd.drive.steering_angle = 0.05
            elif dist > distwant:
                self.cmd.drive.steering_angle = 0
            else:
                ang = disi-180
                self.cmd.drive.steering_angle = ang/600.0
        else:
            self.cmd.drive.steering_angle = 0.3

    #colorval=[[ 52, 66, 61],[ 98, 255, 148]]
    def cd_color_segmentation(img, colorval, show_image=False):
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
        if self.size_calc(box)>1000:
            return True
        return False

    def redbox(self,img):
        box=cd_color_segmentation(img,colorval=[[118, 141, 118],[130, 255, 192]])#[21, 129, 63],[83, 255, 205]  [ 17, 175, 23],[ 82, 255, 148]
        if box[0][0] < 50 and self.size_calc(box)>400:
            print "red stuff"
            self.cmd.drive.steering_angle = (336+50-(box[0][0]+box[1][0])/2)/100.0
        else
            self.left_wall_follow(0.4)

    def theblues(self,img):
        box=cd_color_segmentation(img,colorval=[[118, 141, 118],[130, 255, 192]])#[21, 129, 63],[83, 255, 205]  [ 17, 175, 23],[ 82, 255, 148]
        if box[0][0] < 50 and self.size_calc(box)>200:
            print "blue stuff"
            self.cmd.drive.steering_angle = (336-(box[0][0]+box[1][0])/2)/100.0
        else
            self.left_wall_follow(0.4)

def main():
    try:
        ic = ARDrive()
        rospy.Rate(100)
        while not rospy.is_shutdown():
            ic.drive_pub.publish(ic.cmd)   
            ic.sound_pub.publish(ic.state)
    except rospy.ROSInterruptException:
        exit()

if __name__ == "__main__":
    main()
