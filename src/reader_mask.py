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

        lic_plate = self.findPlate(cameraImage)

        if (lic_plate is not None):
            print("run CNN")
            pos, plate = self.readPlate(lic_plate)
            print("in P{}, plate = {}".format(pos, plate))


    #Uses homography to find the plate
    def findPlate(self, cameraImage):

        # img = cv2.imread('/home/fizzer/ros_ws/src/enph353_robot_controller/reader_utils/ref2.jpg', cv2.IMREAD_GRAYSCALE)
        
        # grayframe = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) #cam image
        image_hsv = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2HSV)
        
        # mask for greys (p4,p5)
        # maskframe = cv2.inRange(image_hsv, np.array([0,0,195],np.uint8), np.array([0,0,210],np.uint8))
        maskframe = cv2.inRange(image_hsv, np.array([120,122,90],np.uint8), np.array([120,255,204],np.uint8))

        mask_h, mask_w = maskframe.shape
        maskframe = maskframe[0:mask_h, 0: int(mask_w/2)]

        # first mask and crop the blues to get where the plate is
        cropped_image = self.defineEdgesandCrop(maskframe, image_hsv)
        cropped_ori = self.defineEdgesandCrop(maskframe, cameraImage)
        # seond mask and crop the cropped image to find the plate
        try:
            crop_mask = cv2.inRange(cropped_image, np.array([0,0,97],np.uint8), np.array([0,0,204],np.uint8))
            final_crop = self.defineEdgesandCrop(crop_mask, cropped_ori)
            h,w,ch = final_crop.shape

            scale = int(350/h)
            if scale < 215/w:
                scale = int(215/w)

            final_crop = cv2.resize(final_crop,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            isitPlate = self.isLicensePlate(final_crop)
            if isitPlate:
                cv2.imshow("cropped", final_crop)
                cv2.waitKey(3)
                return final_crop

        except Exception as e:
            print("oops")

        # cv2.imshow("cam", cameraImage)
        # cv2.waitKey(3)
        # cv2.imshow("mask", maskframe)

        return None

    #goes through our CNN to read the parking spot and read plate
    def readPlate(self, img):
        plate = ""
        pos = ""

        h,w,ch = img.shape
        parking_pic = img[0:200, w-125:w-10] # must be 200 x 100
        pos_pred = self.park_model.predict(parking_pic)
        pos = self.int_to_park[np.argmax(pos_pred)]


        lics_plate = img [h-110:h, 0:w] #shud result in 110 x 215
        scale = int(330/h)
            if scale < 645/w:
                scale = int(645/w)

        lics_plate = cv2.resize(lics_plate,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        for index in range(4):
            if (index <2 ):
                w1 = 30 + (index)*120
            else:
                w1 = 330 + (index - 2)*120
            w2 = w1 + 115
            cropped_img = img[100:255, w1:w2]
            y_predict = self.plate_model.predict(cropped_img)[0]
            plate = plate + self.int_to_char[np.argmax(y_predict)].upper()

        return pos, plate

#### helper method
    def isLicensePlate(self, crop_image):
        img = cv2.imread('/home/fizzer/ros_ws/src/enph353_robot_controller/reader_utils/reference.jpg', cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(img,None)
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # The following code is derived from lab 4
        grayframe = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) #cam image
        

        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe,None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = []

        for m,n in matches: #m is query image, n in image in cam image
            if m.distance < 0.8*n.distance:
                good_points.append(m)

        image_match = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
        cv2.imshow("matches", image_match)
        cv2.waitKey(3)

        if len(good_points) > 10:
            return True

        return False

    def defineEdgesandCrop(self,mask, original):
        dst = cv2.cornerHarris(mask,20,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(mask,np.float32(centroids),(5,5),(-1,-1),criteria)

        max_h = 0
        max_w = 0
        min_h = 1000
        min_w = 1000
        #print(corners.shape)
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
        
        crop = original[min_h-10:max_h+10, min_w-15:max_w+15]
        return crop

def main():
        rospy.init_node("license_read")
        licenseReader = LicenseReader()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rate.sleep()
        
if __name__ == '__main__':
        main()
        