#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from time import sleep

class RobotDrive():

    def __init__(self):

        self.bridge = CvBridge()
        self.imageSubscriber = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.cameraCallback)
        self.cmdVelPublisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.licensePlatePublisher = rospy.Publisher('/license_plate', String, queue_size=10)
        self.cmdVelRate = rospy.Rate(10)
        self.prevError = 0
        self.twist = Twist()
        # All time values are in seconds
        self.timeElapsed = 0
        self.startTime = 0
        self.timeLimit = 240
        self.timeNotInitialized = True
        

    def pid(self, cX, width):
        
        Kp = 2.8*10**(-3)            #2.8*10**(-3)
        Kd = 12*10**(-3)            #13*10**(-3)
        if (cX != -1):
            error = width/2 - cX
            P = Kp * error
            D = Kd * (error - self.prevError)
            self.twist.linear.x = 0.15
            self.twist.angular.z = P + D
            self.prevError = error
        else: 
            self.twist.linear.x = 0
            self.twist.angular.z = 0.5

    def cameraCallback(self, data):

        self.timeElapsed = rospy.get_time() - self.startTime
        
        # Publish initial license plate message
        if (self.timeNotInitialized):
            self.startTime = rospy.get_time()
            self.timeNotInitialized = False
            self.licensePlatePublisher.publish('TeamName,blahblahblah,0,ST00')
   
        elif (self.timeElapsed > 0 and self.timeElapsed <= self.timeLimit):

            try:
                cameraImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            # The following code is derived from Lab 2
            height, width, channels = cameraImage.shape
        
            image_hsv = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2HSV)
            image_rgb = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY)
        
            road_image = image_hsv[height-60:height, 0:width]
            crosswalk_img = image_hsv[height-60:height, 0:width]
            pedestrian_img = image_hsv[height-350:height-200, 0:width]

            road_upperBound = np.array([0, 0, 95], np.uint8)
            road_lowerBound = np.array([0, 0, 65], np.uint8)

            crosswalk_upperBound = np.array([0, 255, 255], np.uint8)
            crosswalk_lowerBound = np.array([0, 255,  220], np.uint8)

            pedestrian_upperBound = np.array([110, 153, 153], np.uint8)
            pedestrian_lowerBound = np.array([100, 63, 63], np.uint8)
            
            road_mask = cv2.inRange(road_image, road_lowerBound, road_upperBound)
            crosswalk_mask = cv2.inRange(crosswalk_img, crosswalk_lowerBound, crosswalk_upperBound)
            pedestrian_mask = cv2.inRange(pedestrian_img, pedestrian_lowerBound, pedestrian_upperBound)
            pedestrian_road_mask = cv2.inRange(pedestrian_img, road_lowerBound, road_upperBound)

            M_road = cv2.moments(road_mask)
            M_crosswalk = cv2.moments(crosswalk_mask)
            M_pedestrian = cv2.moments(pedestrian_mask)
            M_pedestrian_road = cv2.moments(pedestrian_road_mask)
            
            # Detecting Road
            if (int(M_road["m00"]) != 0):
                cX_road = int(M_road["m10"] / M_road["m00"])
                cY_road = int(M_road["m01"] / M_road["m00"])
                print("Found road!")
            else:
                cX_road = -1
                print("No line found!") 

            # Detecting Crosswalk Stop Line
            if (int(M_crosswalk["m00"]) != 0):
                cX_crosswalk = int(M_crosswalk["m10"] / M_crosswalk["m00"])
                print("Found crosswalk!")
            else:
                cX_crosswalk = -1
            
            # Detecting Pedestrian
            if (int(M_pedestrian["m00"]) != 0):
                cX_ped = int(M_pedestrian["m10"] / M_pedestrian["m00"])
                cY_ped = int(M_pedestrian["m01"] / M_pedestrian["m00"])
                print("Found pedestrian!")
            else:
                cX_ped = -1 

            # Determining Pedestrian Road Bounds
            height_road, width_road = pedestrian_road_mask.shape
            b_pedestrian_road = pedestrian_road_mask[height_road-1:height_road, 0:width].tolist()
            b_pedestrian_road = b_pedestrian_road[0]
            pedestrian_crossing = False
            try:
                left_ped_road = b_pedestrian_road.index(255) - 20
                right_ped_road = len(b_pedestrian_road) - 1 - b_pedestrian_road[::-1].index(255) + 10
                
                # Determining if Pedestrian is on the Road
                pedestrian_crossing = True if (cX_ped >= left_ped_road and cX_ped <= right_ped_road) else False
            except ValueError:
                print("No road seen ahead")
            if (cX_crosswalk == -1):
                # PID ALGORITHM
                self.pid(cX_road, width)

            else:
                if (pedestrian_crossing):
                    if ((cX_ped >= width/2+10 or cX_ped <= width/2-10) and cX_ped > 0):
                        self.twist.angular.z = 0
                        self.twist.linear.x = 0
                        print("Pedestrian is crossing!")
                    else:
                        self.pid(cX_road, width)
                else:
                    self.pid(cX_road, width)

            self.cmdVelPublisher.publish(self.twist)
            self.cmdVelRate.sleep()
        
        else:
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            self.cmdVelPublisher.publish(self.twist)
            self.licensePlatePublisher.publish('TeamName,blahblahblah,-1,EN99')
            print('Time Elapsed')
    
    
    
def main():

    rospy.init_node("line_follow")
    lineFollower = RobotDrive()
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        rate.sleep()
    
if __name__ == '__main__':
    main()
