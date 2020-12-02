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
        self.ordered_data_1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.int_to_char = dict((i,c) for i,c in enumerate(self.ordered_data_1))
        self.ordered_data_2 = '123456'
        self.int_to_park = dict((i,c) for i,c in enumerate(self.ordered_data_2))

        # load the parking CNN
        self.sess = tf.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()

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
            pos, plate = self.readPlate(lic_plate)

            if (plate != "" and pos != ""):
                message = "Team7,chuck," + pos +"," + plate
                print("in P{}, plate = {}".format(pos, plate))
                self.ReadPublisher.publish(message)

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
            # print ("No contour detected")
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plate = ""
        pos = ""

        ############ predicting the parking position #################
        # resize the plate
        hi,wi,chi = img.shape
        scale_h = 400/hi
        scale_w = 350/wi
        img = cv2.resize(img,None,fx=scale_h, fy=scale_w, interpolation = cv2.INTER_CUBIC)

        hi,wi,chi = img.shape
        parking_pic = img[int(0.75*hi)-100:int(0.75*hi),wi-150:wi-10]
        parking_pic = parking_pic/255.    
        park_aug = np.expand_dims(parking_pic, axis=0)

        with self.graph.as_default():
            set_session(self.sess)
            pos_pred = self.parkModel.predict(park_aug)[0]
            if np.amax(pos_pred) > 0.5:
                pos = self.int_to_park[np.argmax(pos_pred)]
        
        ############## predicting the license plate ####################
        x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        x = cv2.inRange(x, np.array([119,100,75],np.uint8), np.array([121,245,215],np.uint8))

        img = cv2.merge((x,x,x))
        # img = img [int(0.8*hi):hi, 0:wi] 
        # hi,wi,chi = img.shape
        # cv2.imshow("img", img)

        lics_plate = img[hi-70:hi-5,0:int(wi/2)]
        h,w,ch = lics_plate.shape
        
        # cv2.imshow("pos", lics_plate)
        letter_one = lics_plate[0:65,int(w/2)-65:int(w/2)]
        letter_two = lics_plate[0:65,int(w/2):int(w/2)+65]

        lics_plate = img[hi-70:hi-5,int(wi/2):wi]
        h,w,ch = lics_plate.shape
        num_one = lics_plate[0:65,int(w/2)-65:int(w/2)]
        num_two = lics_plate[0:65,int(w/2):int(w/2)+65]

        # letter_one = letter_one/255.
        # letter_two = letter_two/255.
        # num_one = num_one/255.
        # num_two = num_two/255.

        l1_aug = np.expand_dims(letter_one, axis=0)
        l2_aug = np.expand_dims(letter_two, axis=0)
        n1_aug = np.expand_dims(num_one, axis=0)
        n2_aug = np.expand_dims(num_two, axis=0)

        
        # cv2.imshow("1", letter_one)
        # cv2.imshow("2", letter_two)
        # cv2.imshow("3", num_one)
        # cv2.imshow("4", num_two)
        cv2.waitKey(3)

        with self.graph.as_default():
            try:
                set_session(self.sess)
                l1_pred = self.plateModel.predict(l1_aug)[0]
                l2_pred = self.plateModel.predict(l2_aug)[0]
                n1_pred = self.plateModel.predict(n1_aug)[0]
                n2_pred = self.plateModel.predict(n2_aug)[0]

                plate = plate + self.int_to_char[np.argmax(l1_pred)] + self.int_to_char[np.argmax(l2_pred)] #adds the letters
                plate = plate + self.int_to_char[np.argmax(n1_pred)] + self.int_to_char[np.argmax(n2_pred)] #adds the numbers
            except Exception as e:
                print("plate not found", e)


        # print(letter_one)
        # print("aug", l1_aug)
        print(pos,plate)

        # check if the plate is in the form [char, char, int, int]
        # for i in range(4):
        #     if i < 2:
        #         if (not plate[i].isalpha()):
        #             return pos, None
        #     else:
        #         if (not plate[i].isdigit()):
        #             return pos, None

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
        