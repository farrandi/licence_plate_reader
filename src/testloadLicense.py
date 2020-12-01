#!/usr/bin/env python

import rospy
import cv2
import numpy as np
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
        # set_session(self.sess)
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
            print("running CNN")
            pos, plate = self.readPlate(lic_plate)
            print("in P{}, plate = {}".format(pos, plate))


    #Uses homography to find the plate
    def findPlate(self, cameraImage):

        # img = cv2.imread('/home/fizzer/ros_ws/src/enph353_robot_controller/reader_utils/ref2.jpg', cv2.IMREAD_GRAYSCALE)
        
        # grayframe = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) #cam image
        image_hsv = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2HSV)
        
        # mask for blues
        maskframe = cv2.inRange(image_hsv, np.array([120,122,90],np.uint8), np.array([120,255,204],np.uint8))

        mask_h, mask_w = maskframe.shape
        maskframe = maskframe[0:mask_h, 0: int(mask_w/2)]

        # first mask and crop the blues to get where the plate is
        cropped_image = self.defineEdgesandCrop(maskframe, image_hsv)
        cropped_ori = self.defineEdgesandCrop(maskframe, cameraImage)
        # second mask and crop the cropped image to find the plate

        # try:
        # mask for greys
        crop_mask = cv2.inRange(cropped_image, np.array([0,0,97],np.uint8), np.array([0,0,204],np.uint8))
        crop_2 = self.defineEdgesandCrop(crop_mask, cropped_ori)
        h2,w2,ch2 = crop_2.shape

        if w2 > h2:
            crop_2 = crop_2[0:h2,(w2 - (h2)):w2]
        final_crop = crop_2

        #rescale the image
        h,w, ch = final_crop.shape

        scale = int(350/h)
        if scale < 215/w:
            scale = int(215/w)
        final_crop = cv2.resize(final_crop,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

        #transform the image
        final_hsv = cv2.cvtColor(final_crop, cv2.COLOR_BGR2HSV)
        final_mask = cv2.inRange(final_hsv, np.array([0,0,97],np.uint8), np.array([0,0,204],np.uint8)) #using gray mask

        # cv2.imshow("cropped", final_crop)
        # cv2.waitKey(3)

        matrix = self.getEdgesandTransform(final_mask)
        h,w, ch = final_crop.shape
        final_crop = cv2.warpPerspective(final_crop, matrix, (w, h))

        #check if image is license plate
        isitPlate = self.isLicensePlate(final_crop)

        if isitPlate:
            cv2.imshow("cropped and warped", final_crop)
            cv2.waitKey(3)
            return final_crop

        # except Exception as e:
        #     print(e)
        

        # cv2.imshow("cam", cameraImage)
        # cv2.waitKey(3)
        # cv2.imshow("mask", maskframe)

        return None

    #goes through our CNN to read the parking spot and read plate
    def readPlate(self, img):
        # plateModel = models.load_model("/home/fizzer/ros_ws/src/my_model")
        # parkModel = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader")

        plate = ""
        pos = ""

        h,w,ch = img.shape
        parking_pic = img[40:240, w-116:w-16] # must be 200 x 100
        cv2.imshow("parking", parking_pic)
        cv2.waitKey(3)

        park_aug = np.expand_dims(parking_pic, axis=0)

        with self.graph.as_default():
            set_session(self.sess)
            pos_pred = self.parkModel.predict(park_aug)[0]
            print(pos_pred)
            pos = self.int_to_park[np.argmax(pos_pred)]
        print(pos)

        lics_plate = img [h-110:h, 0:w] #shud result in 110 x 215
        cv2.imshow("license", lics_plate)
        cv2.waitKey(3)

        # scale = int(330/h)+1
        # if scale < 645/w:
        #     scale = int(645/w)+1
        # lics_plate = cv2.resize(lics_plate,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

        for index in range(4):
            if (index <2 ):
                w1 = 15 + (index)*46
            else:
                w1 = 153 + (index - 2)*46
            w2 = w1 + 46
            cropped_img = lics_plate[0:62, w1:w2]
            cropped_img_aug = np.expand_dims(cropped_img, axis=0)
            cv2.imshow("crop", cropped_img)

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
            if m.distance < 0.8*n.distance:
                good_points.append(m)

        image_match = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
        # cv2.imshow("matches", image_match)
        # cv2.waitKey(3)

        if len(good_points) > 10:
            return True

        return False
    
    def getEdgesandTransform(self,image, num = None):
        if num == None:
            num = 5
        dst = cv2.cornerHarris(image,5,3,0.04)
        ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(image,np.float32(centroids),(5,5),(-1,-1),criteria)
        
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

        h,w = image.shape
        
        pts1 = np.float32([[min_w - 5, min_h - 5], [min_w - 5, max_h + 5], [max_w + 5, max_h + 5], [max_w + 5, min_h - 5]])
        pts2 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2) 
        
        return matrix

    def defineEdgesandCrop(self,mask, original, num = None):
        if num is None:
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
        
        crop = original[min_h-10:max_h+20, min_w-15:max_w+15]
        return crop

def main():
        rospy.init_node("license_read")
        licenseReader = LicenseReader()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rate.sleep()
        
if __name__ == '__main__':
        main()
        