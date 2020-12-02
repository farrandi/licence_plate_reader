#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from scipy.spatial import distance as dist
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

        # self.parkModel = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader.h5")
        # self.parkModel._make_predict_function()

        # self.plateModel = models.load_model("/home/fizzer/ros_ws/src/my_model.h5")
        # self.plateModel._make_predict_function()

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

        # if (lic_plate is not None):
        #     pos, plate = self.readPlate(lic_plate)
        #     print("in P{}, plate = {}".format(pos, plate))


    #Uses homography to find the plate
    def findPlate(self, cImage):

        # The following code is from https://medium.com/programming-fever/license-plate-recognition-using-opencv-python-7611f85cdd6c
        # with very slight modifications
        h, w, c = cImage.shape
        cameraImage = cImage[h/2-50:h+50, 0:w/2]
        gray = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY) 
        hsv_im = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv_im, np.array([0,0,97],np.uint8), np.array([0,0,204],np.uint8))

        # cv2.imshow("gray", mask)
        # cv2.waitKey(3)
        
        x, contours, y = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.010 * peri, True)
            area = cv2.contourArea(c)
            
            if len(approx) == 4 and area > 10000:
                max_h = 0
                min_h = 10000
                min_w = 10000
                max_w = 0
                for pts in approx: #approx gives (w,h)
                    h = pts[0,1]
                    w = pts[0,0]
                    point = [w,h]
                    if h > max_h:
                        max_h = h
                    if h < min_h:
                        min_h = h
                    if w > max_w:
                        max_w = w
                    if w < min_w:
                        min_w = w
                w = max_w - min_w
                h = max_h - min_h

                if abs(w-h) < 35:
                    for pts in approx:
                        h = pts[0,1]
                        if max_h - h < 20:
                            pts[0,1] = h + 32

                    screenCnt = approx
                    break

        if screenCnt is None:
            detected = 0
            print ("No contour detected")
        else:
            detected = 1

        if detected == 1:
            # cv2.drawContours(cameraImage, [screenCnt], -1, (0, 0, 255), 3)

            mask = np.zeros(gray.shape,np.uint8)
            new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
            new_image = cv2.bitwise_and(cameraImage, cameraImage,mask=mask)

            # cv2.imshow("new image", new_image)
            # cv2.waitKey(3)
            
            max_h = 0
            min_h = 10000
            min_w = 10000
            max_w = 0

            pts1 = []
            for pts in screenCnt:
                h = pts[0,1]
                w = pts[0,0]
                point = [w,h]
                if h > max_h:
                    max_h = h
                if h < min_h:
                    min_h = h
                if w > max_w:
                    max_w = w
                if w < min_w:
                    min_w = w
                pts1.append(point)

            pts1 = np.array(pts1)
            pts1 = self.order_points(pts1)
            pts2 = np.float32([[min_w, min_h], [min_w, max_h], [max_w, max_h], [max_w, min_h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 

            h,w,ch = new_image.shape
            final_crop = cv2.warpPerspective(new_image, matrix, (w, h))
            final_crop = final_crop[min_h:max_h,min_w:max_w]

            cv2.imshow("cropped", final_crop)
            cv2.waitKey(3)
            return final_crop

        return None

    #goes through our CNN to read the parking spot and read plate
    def readPlate(self, img):
        plate = ""
        pos = ""

        h,w,ch = img.shape
        parking_pic = img[int((0.8*h)-100) : int(0.8*h), w-60:w-10] # must be 100 x 50
        parking_pic = cv2.resize(parking_pic,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        # cv2.imshow("parking", parking_pic)
        # cv2.waitKey(3)
        
        park_aug = np.expand_dims(parking_pic, axis=0)

        with self.graph.as_default():
            set_session(self.sess)
            pos_pred = self.parkModel.predict(park_aug)[0]
            pos = self.int_to_park[np.argmax(pos_pred)]
        

        lics_plate = img [int(0.8*h):h, 0:w] #shud result in 110 x 215
        # scale = int(330/h)+1
        # if scale < 645/w:
        #     scale = int(645/w)+1
        # lics_plate = cv2.resize(lics_plate,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

        h,w,ch = lics_plate.shape
        for index in range(4):
            if (index <2 ):
                w1 = 10 + (index)*30
            else:
                w1 = 100 + (index - 2)*30
            w2 = w1 + 30
            cropped_img = lics_plate[0:h, w1:w2]
            cropped_img_aug = np.expand_dims(cropped_img, axis=0)
            # cv2.imshow("crop", cropped_img)
            # print(cropped_img.shape)

            with self.graph.as_default():
                try:
                    set_session(self.sess)
                    y_pred = self.plateModel.predict(cropped_img_aug)[0]
                    plate = plate + self.int_to_char[np.argmax(y_pred)].upper()
                except Exception as e:
                    print("plate not found", e)
                
        return pos, plate


    ############helper method

    #taken from: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    def order_points(self, pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, bl, br, tr], dtype="float32")

def main():
        rospy.init_node("license_read")
        licenseReader = LicenseReader()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rate.sleep()
        
if __name__ == '__main__':
        main()
        