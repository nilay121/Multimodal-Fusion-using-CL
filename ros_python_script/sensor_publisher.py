# !usr/bin/env python3
## it indicates that this is a python source file

import rospy
import numpy as np
from std_msgs.msg import String, Float32MultiArray
import serial

def read_from_arduino(serial_port):
    try:
        if serial_port.in_waiting > 0:
            line = serial_port.readline().decode('utf-8').rstrip()
            return line
    except Exception as e:
        rospy.logerr("Error reading from Arduino: {}".format(e))
    return None

## create the name of the publisher node 
publisherNodeName = "tactile_sensor_publisher"
topicName = "tactile_topic"
frequency = 20 # 20 hz
baudrate = 9600
port = '/dev/ttyACM0'

rospy.init_node(publisherNodeName, anonymous=True)
publisher = rospy.Publisher(topicName, String, queue_size=60)

## specify the rate/frequency of transmission
rate = rospy.Rate(frequency)
serial_port = serial.Serial(port, baudrate, timeout=1)  # Adjust the serial port as needed
## an imfinite loop that cpatures the images and transmits them through the topic

while not rospy.is_shutdown():
    sensor_value = read_from_arduino(serial_port)
    if sensor_value is not None and sensor_value!="":
        sensor_value = sensor_value.split(",")
        rospy.loginfo(f"Publishing: {sensor_value}")
        publisher.publish(str(sensor_value))
    # we wait for certain amount of time to make sure that the specified transmission rate is achieved
    rate.sleep()

