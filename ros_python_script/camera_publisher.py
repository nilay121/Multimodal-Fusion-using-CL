# !usr/bin/env python3
## it indicates that this is a python source file

import rospy
from sensor_msgs.msg import Image
## we will send messages in trhe form of images so we need to import image
from cv_bridge import CvBridge
## cv bridge package conversts the opencv images into ros image message and vice-versa, and
## thus serves as a bridge between opencv and ros
import cv2 
## create the name of the publisher node 
publisherNodeName = "camera_sensor_publisher"

## create the name of the topic over which we will transmit the image messages
## same name shoul dbe used in the source file of the subscriber
topicName = "video_topic"

## initialize the node
rospy.init_node(publisherNodeName, anonymous=True)
## anonymous parameter= True means that ros will add a random number to publisher node name in order to avoid name conflicts
## it will make sure that we have a unique name for our publisher

## create a publisher object, specify the name of the topic, a type of the message being sent (Image), and the buffer size (queue size)
publisher = rospy.Publisher(topicName, Image, queue_size=60)

## specify the rate/frequency of transmission
rate = rospy.Rate(20)

## Enter the camera id
## 0 --> default
## 2 --> external
camera_id = 2

## create the video capture object
videoCaptureObject = cv2.VideoCapture(camera_id)

## change the resolution of the video
width = 640
height = 480
videoCaptureObject.set(cv2.CAP_PROP_FRAME_WIDTH, width)
videoCaptureObject.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

## create the CvBridge object that will be used to convert openCV images to ROS image messages
bridgeObject = CvBridge()

## ------------------------------------------- vision part -------------------------------------------

## an imfinite loop that cpatures the images and transmits them through the topic
while not rospy.is_shutdown():
    ## retuens two values, the first value is the boolean for success/failure
    ## the second is the actual frame
    returnValue, capturedFrame = videoCaptureObject.read()
    ## No error then transmit
    if returnValue == True:
        ## print messgae
        rospy.loginfo("Video frame captured and published")
        # convert opencv to ros image message
        imageToTransmit = bridgeObject.cv2_to_imgmsg(capturedFrame)
        # publish the converted image through the topic
        publisher.publish(imageToTransmit)
    # we wait for certain amount of time to make sure that the specified transmission rate is achieved
    rate.sleep()

    if rospy.is_shutdown():
        videoCaptureObject.release()
        break
