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
        self.cmdVelPublisher = rospy.Publisher('/license_plate', String, queue_size = 10)
        self.cmdVelRate = rospy.Rate(10)
        self.prevError = 0

        #load the trained CNN for reading licence plate
        self.ordered_data_1 = 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.int_to_char = dict((i,c) for i,c in enumerate(self.ordered_data_1))
        self.plate_model = models.load_model("/home/fizzer/ros_ws/src/my_model")

        #load the parking CNN
        self.ordered_data_2 = '123456789'
        self.int_to_park = dict((i,c) for i,c in enumerate(self.ordered_data_2))
        self.park_model = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader")

        # set reference.jpg as query image 
        # self.img = cv2.imread('/home/fizzer/ros_ws/src/enph353_robot_controller/reader_utils/reference.jpg', cv2.IMREAD_GRAYSCALE)


        # Feature matching
        # self.index_params = dict(algorithm=0, trees=5)
        # self.search_params = dict()
        # self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    # This is the Main read code
    def findandread(self, data):
        try:
            cameraImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        status, hom_img = self.findPlate(cameraImage)
        while status:
            # plate = self.readPlate(hom_img)
            # print(plate)
            print("looping")
            status = input("type False to stop looping")

    #Uses homography to find the plate
    def findPlate(self, cameraImage):

        img = cv2.imread('/home/fizzer/ros_ws/src/enph353_robot_controller/reader_utils/reference.jpg', cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(img,None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # The following code is derived from lab 4
        grayframe = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) #cam image

        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe,None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = []

        for m,n in matches: #m is query image, n in image in cam image
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        image_match = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
        cv2.imshow("matches", image_match)

        # Homography
        if len(good_points) > 5:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            macthes_mask = mask.ravel().tolist()

            # perspective transform
            h,w = img.shape
            pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(cameraImage, [np.int32(dst)], True, (255,0,0), 3)
            
            cv2.imshow("Homography", homography)
            cv2.waitKey(3)
            print("homography: FOUND")
            return True, homography
        else:
            # cv2.imshow("Homography", grayframe)
            return False, cameraImage

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
        