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

        #load the trained CNN from lab 5
        self.ordered_data = 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.int_to_char = dict((i,c) for i,c in enumerate(self.ordered_data))

        self.loaded_model = models.load_model("/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts/reader_utils/my_model")

        # self.json_file = open('/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts/reader_utils/model.json', 'r')
        # self.loaded_model_json = self.json_file.read()
        # self.json_file.close()
        # self.loaded_model = models.model_from_json(self.loaded_model_json)
        # self.loaded_model.load_weights("/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts/reader_utils/model.h5")
        # print("Loaded model from disk")

        # set blank+plate.png as query image 
        self.img = cv2.imread('/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts/reader_utils/blank_plate.png', cv2.IMREAD_GRAYSCALE)

        #features
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img,None)

        # Feature matching
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def findPlate(self, data):
        try:
            cameraImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # The following code is derived from lab 4
        grayframe = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) #train image
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe,None)
        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)
        good_points = []

        for m,n in matches: #m is query image, n in image in train image
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        # Homography
        if len(good_points) > 8:
            query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)

            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            macthes_mask = mask.ravel().tolist()

            # perspective transform
            h,w = self.img.shape
            pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts, matrix)

            homography = cv2.polylines(cameraImage, [np.int32(dst)], True, (255,0,0), 3)

            return True, homography
            cv2.imshow("Homography", homography)
            cv2.waitKey(3)
        else:
            # cv2.imshow("Homography", grayframe)
            return False, cameraImage

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
            y_predict = conv_model.predict(cropped_img)[0]
            plate = plate + self.int_to_char[np.argmax(y_predict)].upper()

        return plate

    def findandread(self,data):
        status, hom_img = self.findPlate(data)
            while status:
                plate = self.readPlate(hom_img)
                print(plate)


def main():
        rospy.init_node("license_read")
        licenseReader = LicenseReader()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rate.sleep()
        
if __name__ == '__main__':
        main()
        