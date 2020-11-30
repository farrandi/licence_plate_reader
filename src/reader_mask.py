#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend

class LicenseReader():

    def __init__(self):

        self.bridge = CvBridge()
        self.imageSubscriber = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.findandread)
        self.ReadPublisher = rospy.Publisher('/license_plate', String, queue_size = 10)
        self.ReadRate = rospy.Rate(10)
        self.prevError = 0

        #load the trained CNN for reading licence plate
        self.ordered_data_1 = 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.int_to_char = dict((i,c) for i,c in enumerate(self.ordered_data_1))
        self.plate_model = models.load_model("/home/fizzer/ros_ws/src/my_model")

        #load the parking CNN
        self.ordered_data_2 = '123456789'
        self.int_to_park = dict((i,c) for i,c in enumerate(self.ordered_data_2))
        self.park_model = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader")

    # This is the Main read code
    def findandread(self, data):
        try:
            cameraImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hom_img = self.findPlate(cameraImage)
        while status:
            print("looping")
            status = input("type False to stop looping")

    #Uses homography to find the plate
    def findPlate(self, cameraImage):

        # img = cv2.imread('/home/fizzer/ros_ws/src/enph353_robot_controller/reader_utils/ref2.jpg', cv2.IMREAD_GRAYSCALE)
        
        # grayframe = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) #cam image
        image_hsv = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2HSV)
        maskframe = cv2.inRange(image_hsv, np.array([0,0,195],np.uint8), np.array([0,0,210],np.uint8))
        mask_h, mask_w = maskframe.shape
        maskframe = maskframe[0:mask_h - 200, 0: int(mask_w/2)]


        #finding the edges
        dst = cv2.cornerHarris(maskframe,25,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(maskframe,np.float32(centroids),(5,5),(-1,-1),criteria)

        max_h = 0
        max_w = 0
        min_h = 1000
        min_w = 1000
        print(corners.shape)
        corners = corners[1:len(corners),:]

        for points in corners:
            if points[0] > max_w:
                max_w = int(points[0])
            if points[0] < min_w:
                min_w = int(points[0])
            if points[1] > max_h:
                max_h = int(points[1])
            if points[1] < min_h:
                min_h = int(points[1])
        

        cropped_image = cameraImage[min_h-10:max_h+10, min_w-10:max_w+10]
        # for i in range(1, len(corners)):
        #     print(corners[i])      

        cv2.imshow("cam", cameraImage)
        cv2.imshow("mask", maskframe)
        cv2.imshow("cropped", cropped_image)
        cv2.waitKey(3)

        return maskframe

    #goes through our CNN to read the parking spot and read plate
    def readPlate(self, homography_im):
        plate = ""

        #resize the homography to: 298 X 600
        

        for index in range(4):
            if (index <2 ):
                w1 = 30 + (index)*120
            else:
                w1 = 330 + (index - 2)*120
            w2 = w1 + 115
            cropped_img = homography_im[0:255, w1:w2]
            y_predict = self.plate_model.predict(cropped_img)[0]
            plate = plate + self.int_to_char[np.argmax(y_predict)].upper()

        return plate


def main():
        rospy.init_node("license_read")
        licenseReader = LicenseReader()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rate.sleep()
        
if __name__ == '__main__':
        main()
        