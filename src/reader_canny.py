#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import imutils
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

class LicenseReader():

    def __init__(self):

        tf.keras.backend.clear_session()
        # define the parking and plate dictionaries
        self.ordered_data_1 = 'abcdefghijklmnopqrstuvwxyz0123456789'
        self.int_to_char = dict((i,c) for i,c in enumerate(self.ordered_data_1))
        self.ordered_data_2 = '123456'
        self.int_to_park = dict((i,c) for i,c in enumerate(self.ordered_data_2))

        # load the parking CNN
        self.sess = tf.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        self.parkModel = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader.h5")
        self.parkModel._make_predict_function()

        self.plateModel = models.load_model("/home/fizzer/ros_ws/src/my_model.h5")
        self.plateModel._make_predict_function()

        self.bridge = CvBridge()
        self.imageSubscriber = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.findandread)
        self.ReadPublisher = rospy.Publisher('/license_plate', String, queue_size = 10)
        self.ReadRate = rospy.Rate(10)
        self.prevError = 0



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
    def findPlate(self, cImage):

        # The following code is from https://medium.com/programming-fever/license-plate-recognition-using-opencv-python-7611f85cdd6c
        # with very slight modifications
        h, w, c = cImage.shape
        cameraImage = cImage[h/2-50:h+50, 0:w/2]
        gray = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) 
        gray = cv2.bilateralFilter(gray, 13, 15, 15) 

        edged = cv2.Canny(gray, 30, 200) 
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.010 * peri, True)
        
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            detected = 0
            print ("No contour detected")
        else:
            detected = 1

        if detected == 1:
            #cv2.drawContours(cameraImage, [screenCnt], -1, (0, 0, 255), 3)

            mask = np.zeros(gray.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(cameraImage, cameraImage,mask=mask)
            # cv2.imshow("mask", mask)
            # cv2.imshow("edges", edged)
            # cv2.imshow("2 mask", crop_2_mask)
            # cv2.imshow("final image", new_image)
            cv2.waitKey(3)


            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = cameraImage[topx:bottomx+25, topy:bottomy+1]

            img = cv2.resize(cameraImage,(500,300))
            cropped = cv2.resize(cropped,(215,350))
            #cv2.imshow('car',cameraImage)

            
            if self.isLicensePlate(cropped):
                cv2.imshow('License Plate',cropped)
                cv2.waitKey(3)
                return cropped

        return None

    #goes through our CNN to read the parking spot and read plate
    def readPlate(self, img):
        # plateModel = models.load_model("/home/fizzer/ros_ws/src/my_model")
        # parkModel = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader")

        plate = ""
        pos = ""

        h,w,ch = img.shape
        parking_pic = img[40:240, w-130:w-30] # must be 200 x 100
        # cv2.imshow("parking", parking_pic)
        # cv2.waitKey(3)

        park_aug = np.expand_dims(parking_pic, axis=0)

        with self.graph.as_default():
            set_session(self.sess)
            pos_pred = self.parkModel.predict(park_aug)[0]
            print(pos_pred)
            pos = self.int_to_park[np.argmax(pos_pred)]
        print(pos)

        lics_plate = img [h-110:h, 0:w] #shud result in 110 x 215
        scale = int(330/h)+1
        if scale < 645/w:
            scale = int(645/w)+1
        lics_plate = cv2.resize(lics_plate,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

        for index in range(4):
            if (index <2 ):
                w1 = 30 + (index)*120
            else:
                w1 = 330 + (index - 2)*120
            w2 = w1 + 115
            cropped_img = lics_plate[100:255, w1:w2]
            cropped_img_aug = np.expand_dims(cropped_img, axis=0)
            # cv2.imshow("crop", cropped_img)

            with self.graph.as_default():
                try:
                    set_session(self.sess)
                    y_pred = self.plateModel.predict(cropped_img_aug)[0]
                    plate = plate + self.int_to_char[np.argmax(y_pred)].upper()
                except Exception as e:
                    print("plate not found")
                
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
            if m.distance < 0.7*n.distance:
                good_points.append(m)

        image_match = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

        cv2.imshow("matches", image_match)
        cv2.waitKey(3)

        if len(good_points) > 10:
            return True

        return False

    def defineEdgesandCrop(self,mask, original, num = None):
        if num == None:
            num = 20
        dst = cv2.cornerHarris(mask,num,3,0.04)
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
        