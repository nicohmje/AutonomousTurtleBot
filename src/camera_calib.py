#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rospy
import yaml
from sensor_msgs.msg import CameraInfo, Image


"""
    Load camera calibration for the simulation. We calibrated the camera manually in gazebo.
"""




class CameraCalib():
    def __init__(self): 
        self.camera_info = None
        rospy.init_node("camera_info_publisher")

        # Parse yaml file
        # Initialize publisher node
        rospy.Subscriber("/camera/image", Image, self.yaml_to_CameraInfo)
        self.publisher = rospy.Publisher("/camera/camera_info_correct", CameraInfo, queue_size=10)

        rospy.spin()
        

    def yaml_to_CameraInfo(self, image):

        """
        Parse a yaml file containing camera calibration data (as produced by 
        rosrun camera_calibration cameracalibrator.py) into a 
        sensor_msgs/CameraInfo msg.
        
        Parameters
        ----------
        yaml_fname : str
            Path to yaml file containing camera calibration data

        Returns
        -------
        camera_info_msg : sensor_msgs.msg.CameraInfo
            A sensor_msgs.msg.CameraInfo message containing the camera calibration
            data
        """

        yaml_fname = rospy.get_param('camera_calib') 

        # Load data from file
        with open(yaml_fname, "r") as file_handle:
            calib_data = yaml.load(file_handle, Loader=yaml.FullLoader)
        # Parse
        camera_info_msg = CameraInfo()
        camera_info_msg.header.stamp = image.header.stamp
        camera_info_msg.header.frame_id = image.header.frame_id
        camera_info_msg.width = calib_data["image_width"]
        camera_info_msg.height = calib_data["image_height"]
        camera_info_msg.K = calib_data["camera_matrix"]["data"]
        camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
        camera_info_msg.R = calib_data["rectification_matrix"]["data"]
        camera_info_msg.P = calib_data["projection_matrix"]["data"]
        camera_info_msg.distortion_model = calib_data["distortion_model"]
        self.publisher.publish(camera_info_msg)
    

if __name__ == "__main__":
    try:
        # Create a LidarProcess and start it
        cminfo = CameraCalib()
    except rospy.ROSInterruptException:
        # If a ROSInterruptException occurs, exit the program
        rospy.logwarn("DOAWIDUAWIUDBAWIDABWIDBW")
        exit(0)