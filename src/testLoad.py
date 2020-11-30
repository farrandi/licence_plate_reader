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

class LoadTest():

    def __init__(self):
        #load the parking CNN
        self.ordered_data_2 = '123456789'
        self.int_to_park = dict((i,c) for i,c in enumerate(self.ordered_data_2))

        tf.keras.backend.clear_session()
        # self.sess = tf.Session()
        # self.graph = tf.get_default_graph()
        self.sess = tf.keras.backend.get_session()
        self.graph = tf.compat.v1.get_default_graph()
        

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        # set_session(self.sess)
        self.parkModel = models.load_model("/home/fizzer/ros_ws/src/my_parking_reader.h5")
        self.parkModel._make_predict_function()

        self.bridge = CvBridge()
        self.imageSubscriber = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.test)
        self.ReadPublisher = rospy.Publisher('/license_plate', String, queue_size = 10)
        self.ReadRate = rospy.Rate(10)
        self.prevError = 0

    def test(self, data):
        try:
            cameraImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        h,w,ch = cameraImage.shape
        parking_pic = cameraImage[0:200, w-110:w-10] # must be 200 x 100
        cv2.imshow("pic",parking_pic)
        cv2.waitKey(3)
        park_aug = np.expand_dims(parking_pic, axis=0)

        pos = "nothing"
        with self.graph.as_default():
            set_session(self.sess)
            pos_pred = self.parkModel.predict(park_aug)[0]
            pos = self.int_to_park[np.argmax(pos_pred)]
        print(pos)


def main():
        rospy.init_node("testing")
        loadtest = LoadTest()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            rate.sleep()
        
if __name__ == '__main__':
        main()
        