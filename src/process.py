#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError


from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import matplotlib.pyplot as plt
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid


import math



class CameraProcess:
    def __init__(self):
            self.image_pub = rospy.Publisher("/masked_frame", Image, queue_size=1)
            self.error_pub = rospy.Publisher("/error", Int32, queue_size=1)
            self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)


            rospy.Subscriber("/param_change_alert", Bool, self.get_params)

            self.cv_image = None
            self.laser_scan = None

            self.detect_stepline = False
            self.step = 0

            self.ang_vel = 0
            self.cmd_speed = 0


            self.min_angle_deg = -90
            self.max_angle_deg = 90

            self.occupancy_grid = None

            self.left_lane = None

            self.LOOKAHEAD = 1 #meters
            self.CELLS_PER_METER = 25
            self.IS_FREE = 0
            self.IS_OCCUPIED = 100


            self.bridge = CvBridge()
            self.get_params()

            rospy.Subscriber("/camera/image", Image, self.callback_image)
            rospy.Subscriber("/occupancy_grid_noroad", OccupancyGrid, self.occupCB)
            rospy.Subscriber("/lidar_data", LaserScan, self.lidarCB)
            rospy.Subscriber("/image_rect_color", Image, self.callback_image_rect)




    def occupCB(self, msg):
        data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.occupancy_grid = np.ma.array(data, mask=data==-1, fill_value=-1)
        self.grid_height = msg.info.width
        self.stamp = msg.header.stamp
        self.grid_width = msg.info.height
        self.CELL_Y_OFFSET = (self.grid_width // 2) - 1


    def make_cmd(self):
        self.cmd_twist = Twist()
        # self.cmd_speed = 0
        # self.ang_vel = 0

        print("Step : ", self.step)
        if not self.step:
            self.cmd_speed = 0.2
            self.ang_vel = 0

        if self.step == 1:

            if self.cv_image is None:
                rospy.logwarn("No image received!")
                pass
            else:
                #pass
                self.lane_assist_controller()

        elif self.step == 2:
            if self.laser_scan is None:
                rospy.logwarn("No LiDAR data received!")
                pass
            else:
                self.lane_assist_controller()
                self.obstacle_detection()
        
        elif self.step == 3:
            if self.laser_scan is None:
                rospy.logwarn("No LiDAR data received!")
                pass
            else:
                self.lidar_only()
                pass


        # print("ang_vel = ", self.ang_vel)
        self.cmd_twist.linear.x = self.cmd_speed
        self.cmd_twist.angular.z = self.ang_vel

        # print(self.cmd_twist)

        #speed pub
        #self.cmd_vel_pub.publish(self.cmd_twist)



    def lidarCB(self, data:LaserScan) :
        self.laser_scan = data

    def lidar_only(self):
        # center_pos_index = int(len(self.laser_scan.ranges)//2 + 1)

        # # print("ang_vel", self.ang_vel)

        # offset = 75

        Kp = 3.0

        # print(90 - np.argmax(self.laser_scan.ranges))

        # dir = Kp * (90 - np.argmax(self.laser_scan.ranges))

        # print("dir ", dir)

        dst_left = np.mean(self.laser_scan.ranges[120:179])
        dst_right = np.mean(self.laser_scan.ranges[0:60])



        dir = Kp * (dst_left-dst_right)

        print(dir)

        print("left ", dst_left)
        print(dst_right)

        self.ang_vel = dir
        self.cmd_speed = 0.1





    def traverse_grid(self, start, end):
        """
        Bresenham's line algorithm for fast voxel traversal

        CREDIT TO: Rogue Basin
        CODE TAKEN FROM: http://www.roguebasin.com/index.php/Bresenham%27s_Line_Algorithm
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        return points
    
    

    def check_collision(self, cell_a, cell_b, margin=0):
        """
        Checks whether the path between two cells
        in the occupancy grid is collision free.

        The margin is done by checking if adjacent cells are also free.

        One of the issues is that if the starting cell is next to a wall, then it already considers there to be a collision.
        See check_collision_loose


        Args:
            cell_a (i, j): index of cell a in occupancy grid
            cell_b (i, j): index of cell b in occupancy grid
            margin (int): margin of safety around the path
        Returns:
            collision (bool): whether path between two cells would cause collision
        """ 
        obstacles = []

        for i in range(-margin, margin + 1):  # for the margin, check
            cell_a_margin = (cell_a[0]+i, cell_a[1])
            cell_b_margin = (cell_b[0]+i, cell_b[1])
            for cell in self.traverse_grid(cell_a_margin, cell_b_margin):
                if (cell[0] * cell[1] < 0) or (cell[0] >= self.grid_height) or (cell[1] >= self.grid_width):
                    continue
                try:
                    if self.occupancy_grid[cell[0], cell[1]] == self.IS_OCCUPIED:
                        obstacles.append(cell)
                        break
                except:
                    print("Out of bounds")
                    obstacles.append(cell)
                    break
        return obstacles



    def crop_data(self, data:list, angle_increment) :

        angle_min_crop = math.radians(self.min_angle_deg)
        angle_max_crop = math.radians(self.max_angle_deg)


        angle_min = -3.1415926535
        angle_max = 3.1415926535
        ranges = np.roll(np.array(data), int(len(data)/2)) #-pi/2 is first

        # end_index = int(round((angle_max_crop*2) / angle_increment))

        # cropped_ranges = ranges[:end_index]

        # Calculate start and end indices for cropping
        start_index = int((angle_min_crop - angle_min) / angle_increment)
        end_index = int((angle_max_crop - angle_min) / angle_increment)

        # Ensure indices are within the range of available data
        start_index = max(0, min(start_index, len(ranges)))
        end_index = max(0, min(end_index, len(ranges)))

        # Crop the range data
        cropped_ranges = ranges[start_index:end_index+1]

        return list(cropped_ranges)



    def get_params(self, event=True):
        rospy.loginfo("Updating the parameters")

        self.sim = rospy.get_param("use_sim_time")

        # Yellow Color Gains
        self.left_H_l = rospy.get_param("/left_H_l", default=77)
        self.left_S_l = rospy.get_param("/left_S_l", default=32)
        self.left_V_l = rospy.get_param("/left_V_l", default=76)

        self.left_H_u = rospy.get_param("/left_H_u", default=102)
        self.left_S_u = rospy.get_param("/left_S_u", default=180)
        self.left_V_u = rospy.get_param("/left_V_u", default=132)

        # White Color Gains
        self.right_H_l = rospy.get_param("/right_H_l", default=100)
        self.right_S_l = rospy.get_param("/right_S_l", default=74)
        self.right_V_l = rospy.get_param("/right_V_l", default=123)

        self.right_H_u = rospy.get_param("/right_H_u", default=120)
        self.right_S_u = rospy.get_param("/right_S_u", default=255)
        self.right_V_u = rospy.get_param("/right_V_u", default=255)

        # Red Color Gains (Assuming two ranges for red)
        self.stepline1_H_l = rospy.get_param("/stepline1_H_l", default=0)
        self.stepline1_S_l = rospy.get_param("/stepline1_S_l", default=0)
        self.stepline1_V_l = rospy.get_param("/stepline1_V_l", default=0)

        self.stepline1_H_u = rospy.get_param("/stepline1_H_u", default=10)
        self.stepline1_S_u = rospy.get_param("/stepline1_S_u", default=255)
        self.stepline1_V_u = rospy.get_param("/stepline1_V_u", default=255)

        self.stepline2_H_l = rospy.get_param("/stepline2_H_l", default=170)
        self.stepline2_S_l = rospy.get_param("/stepline2_S_l", default=50)
        self.stepline2_V_l = rospy.get_param("/stepline2_V_l", default=131)

        self.stepline2_H_u = rospy.get_param("/stepline2_H_u", default=180)
        self.stepline2_S_u = rospy.get_param("/stepline2_S_u", default=170)
        self.stepline2_V_u = rospy.get_param("/stepline2_V_u", default=222)




    def callback_image(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_image(self.cv_image)
        

    def callback_image_rect(self, msg):
        self.cv_image_rect = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.second_process(self.cv_image_rect)

    # def local_to_grid(self, x, y):
    #     i = int(x * -self.CELLS_PER_METER + (self.grid_height - 1))
    #     j = int(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET)
    #     return (i, j)

    # def local_to_grid_parallel(self, x, y):
    #     i = np.round(x * -self.CELLS_PER_METER + (self.grid_height - 1)).astype(int)
    #     j = np.round(y * -self.CELLS_PER_METER + self.CELL_Y_OFFSET).astype(int)
    #     return i, j

    # def grid_to_local(self, point):
    #     i, j = point[0], point[1]
    #     x = (i - (self.grid_height - 1)) / -self.CELLS_PER_METER
    #     y = (j - self.CELL_Y_OFFSET) / -self.CELLS_PER_METER
    #     return (x, y)


    def obstacle_detection(self):

        if self.occupancy_grid is None:
            return False
        # print("Obstacles left front right ", self.obs_left, self.obs_front, self.obs_right)
        
        MARGIN = 2
        current_pos = (12,0)
        goal_pos = (12,5)


        error = 12 - (np.mean(self.check_collision(current_pos, goal_pos, margin=MARGIN), axis = 0))

        # print("error", error)

        if (np.isnan(error)).any():
            pass
        else:
            print("Obstacle!", error[0])
            self.ang_vel = error[0] * 2.0




    def lane_assist_controller(self):

        if self.left_lane is None:
            return

        rows, cols = self.image.shape[:2]

        speed = 0.2

        # print("left, right", self.left_lane, self.right_lane)


        if self.left_lane[1]> 320:
            self.left_lane[1] = 0
        if self.right_lane[1]<320:
            self.right_lane[1] = 640

        # if np.isnan(self.left_lane[1]) :
            # print("left is nan")


        # if np.isnan(self.right_lane[1]):
            # print("right is nan")

        if np.isnan(self.left_lane[1]):
            if np.isnan(self.right_lane[1]):
                # print("left and right nan")
                center_of_road = self.image.shape[0]//2
                speed = 0
            else:
                # print("left nan")
                center_of_road = self.right_lane[1] - (cols * 0.45)
        else:
            if np.isnan(self.right_lane[1]):
                # print("right nan")
                center_of_road = self.left_lane[1] + (cols * 0.45)
            else:
                center_of_road = self.left_lane[1] + (self.right_lane[1] - self.left_lane[1])*0.5

        # print("width ", self.right_lane[1] - self.left_lane[1])

        if (self.right_lane[1] - self.left_lane[1]) > 550:
            print("Road split detected, following yellow line.")
            print("left lane, cols", self.left_lane[1], cols*0.45)
            center_of_road = self.left_lane[1] + (cols * 0.45)


        self.error = center_of_road - self.image.shape[1]//2

        # print("error", self.error)

        # print("img shape", self.image.shape)

        # print("center of road", center_of_road)

        try:
            self.error_pub.publish(Int32(int(self.error)))
        except:
            self.error_pub.publish(Int32(0))

        center_left = tuple(np.intp(self.left_lane))  # Convert centroid to integer tuple
        center_left = tuple([center_left[1], center_left[0]])
        center_right = tuple(np.intp(self.right_lane))  # Convert centroid to integer tuple
        center_right = tuple([center_right[1], center_right[0]])


        # center_road = tuple([int(center_of_road), 210])

        # radius = 10  # Smaller radius for the circle we're drawing
        # color = (0, 255, 0)  # Green color
        # color2 = (255, 255, 0)  # Yellow color
        # thickness = 2  # Thickness of the circle outline

        # print("cetner road ", center_road)

        # Draw the circle on the self.image
        # cv.circle(self.image, center_left, radius, color, thickness)
        # cv.circle(self.image, center_right, radius, color, thickness)
        # cv.circle(self.image, center_road, radius, color2, 5)




        self.cmd_speed =  speed #0.22
        self.ang_vel = self.error * -0.025#0.03

    #Function that warps the image
    def warp(self, img, source_points, destination_points, destn_size):
        matrix = cv.getPerspectiveTransform(source_points, destination_points)
        warped_img = cv.warpPerspective(img, matrix, destn_size)
        return warped_img

    #Function that unwarps the image
    def unwarp(self, img, source_points, destination_points, source_size):
        matrix = cv.getPerspectiveTransform(destination_points, source_points)
        unwarped_img = cv.warpPerspective(img, matrix, source_size)
        return unwarped_img

    def preprocessing(self, img):

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        lower_yellow = np.array([77,0,94])
        upper_yellow = np.array([102,97,190])

        # Define HSV thresholds for red color
        lower_red1 = np.array([0, 0, 00])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 50,131])
        upper_red2 = np.array([180, 170, 222])



		# Create masks for yellow and white colors
        mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
        mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
        masked_red_inter = cv.bitwise_or(mask_red1, mask_red2, mask=mask_red2)


		# Bitwise AND masks with original image


        # masked_yellow = cv.bitwise_and(hsv, hsv, mask=mask_yellow)
        # masked_white = cv.bitwise_and(hsv, hsv, mask=masked_red_inter)

		# Combine masked images
        masked_frame = mask_yellow + masked_red_inter

        return masked_frame

    #Function that defines the polygon region of interest
    def regionOfInterest(self,img, polygon):
        mask = np.zeros_like(img)
        x1, y1 = polygon[0]
        x2, y2 = polygon[1]
        x3, y3 = polygon[2]
        x4, y4 = polygon[3]
        m1 = (y2-y1)/(x2-x1)
        m2 = (y3-y2)/(x3-x2)
        m3 = (y4-y3)/(x4-x3)
        m4 = (y4-y1)/(x4-x1)
        b1 = y1 - m1*x1
        b2 = y2 - m2*x2
        b3 = y3 - m3*x3
        b4 = y4 - m4*x4

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if i>=m1*j+b1 and i>=m2*j+b2 and i>=m3*j+b3 and i<=m4*j+b4:
                    mask[i][j] = 1

        masked_img = np.multiply(mask, img)
        return masked_img

    def fitCurve(self,img):
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows = 50
        margin = 50
        minpix = 50
        window_height = int(img.shape[0]/nwindows)
        y, x = img.nonzero()
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_indices = []
        right_lane_indices = []

        for window in range(nwindows):
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
            good_right_indices  = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]
            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)
            if len(good_left_indices) > minpix:
                leftx_current = int(np.mean(x[good_left_indices]))
            if len(good_right_indices) > minpix:
                rightx_current = int(np.mean(x[good_right_indices]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)
        leftx = x[left_lane_indices]
        lefty = y[left_lane_indices]
        rightx = x[right_lane_indices]
        righty = y[right_lane_indices]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    #Function that give pixel location of points through which the curves of detected lanes passes
    def findPoints(self,img_shape, left_fit, right_fit):
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        return pts_left, pts_right

    #Function that fills the space between the detected lane curves
    def fillCurves(self,img_shape, pts_left, pts_right):
        pts = np.hstack((pts_left, pts_right))
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype='uint8')
        cv.fillPoly(img, np.int_([pts]), (0,255, 0))
        return img

    def radiusOfCurvature(self,img, left_fit, right_fit):
        y_eval = img.shape[0]//2
        left_radius = -1 * ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / (2*left_fit[0])
        right_radius = -1 * ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / (2*right_fit[0])
        avg_radius = (left_radius+right_radius)/2
        return round(left_radius,2), round(right_radius,2), round(avg_radius,2)

    def combine_vision_and_lidar(self, roadmap):
        occup_grid = self.occupancy_grid

        pass

    def second_process(self, image):
        processed_img = self.preprocessing(image)


        height, width = processed_img.shape

        # polygon = [(int(width*0.05), int(height)), (int(width*0.3), int(height*0.7)), (int(width*0.7), int(height*0.7)), (int(0.95*width), int(height))]
        polygon = [(int(0), int(height)), (int(1), int(158)), (int(width-1), int(158)), (int(width), int(height))]
        masked_img = self.regionOfInterest(processed_img, polygon)


        source_points = np.float32([[0,0], [width,0], [-850,height], [width+850,height]])

        destination_points = np.float32([[0,0], [500,0], [0,600],[500, 600]])
        warped_img_size = (500, 600)
        warped_img = self.warp(processed_img, source_points, destination_points, warped_img_size)
        kernel = np.ones((11,11), np.uint8)
        opening = cv.morphologyEx(warped_img, cv.MORPH_CLOSE, kernel)

        warped_img_shape = (warped_img.shape)

        left_fit, right_fit = self.fitCurve(opening)
        pts_left, pts_right = self.findPoints(warped_img_shape, left_fit, right_fit)
        fill_curves = self.fillCurves(warped_img_shape, pts_left, pts_right)


        # self.combine_vision_and_lidar(fill_curves)
        center_point = pts_left[0][-1] + (pts_right[0][-1] - pts_left[0][0])*0.5
        waypoint = pts_left[0][0] + (pts_right[0][0] - pts_left[0][-1])*0.5

        self.heading_error = waypoint - warped_img.shape[1]//2
        self.crosstrack_error = center_point - warped_img.shape[1]//2

        center_point = tuple(np.intp(center_point))
        waypoint = tuple(np.intp(waypoint))

        radius = 10  # Smaller radius for the circle we're drawing
        color = (0, 0, 255)  # Green color
        color2 = (255, 0, 0)  # Green color
        thickness = 2  # Thickness of the circle outline

        # Draw the circle on the imageq
        cv.circle(fill_curves, center_point, radius, color, thickness)
        cv.circle(fill_curves, waypoint, 2, color2, 50)


        # print(fill_curves.shape)

        unwarped_fill_curves = self.unwarp(fill_curves, source_points, destination_points, (width, height))
        window1 = cv.addWeighted(image, 1, unwarped_fill_curves, 1, 0)
        left_radius, right_radius, avg_radius = self.radiusOfCurvature(warped_img, left_fit, right_fit)



        if abs(left_radius)>5000 and abs(right_radius)>5000:
            # print("road is straight")
            pass
            # self.error = 0
        elif abs(avg_radius)<5000:
            if avg_radius < 0:
                pass
                # print("Road is turning right ", avg_radius)
            else:
                pass
                # print("Road is turnign left ", avg_radius)
            # self.error = (5000. - avg_radius)/5000.



        # self.cmd_speed =  0.22 #0.22
        # self.ang_vel = (self.crosstrack_error[0] * -0.01) + self.heading_error[0] * -0.01

        # print(self.ang_vel)
        # print("Heading error ", self.heading_error[0])
        # print("Crosstrack error ", self.crosstrack_error[0])
        pass
        
    def undistort(self, img, balance=0.3, dim2=None, dim3=None):
        DIM = [320,240]
        K = np.array([[95.06302, 0.04031, 159.22853],
              [0., 95.154, 119.47541],
              [0., 0., 1.]])  # Camera matrix
        D = np.array([0.017580, 0.003274, -0.000498, -0.000042])  # Distortion coefficients

        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv.CV_16SC2)
        undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        return undistorted_img



    def process_image(self, image):

        self.image = np.copy(image)

        rectangle = np.copy(image)


        if self.sim:
            rectangle[:150, :, :] = 0
            rectangle[220:, :, :] = 0
        else:
            rectangle[:, :40, :] = 0 
            rectangle[:, 280:, :] = 0

        # Convert image to HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)


        # Left lane
        lower_left = np.array([self.left_H_l, self.left_S_l, self.left_V_l])
        upper_left = np.array([self.left_H_u, self.left_S_u, self.left_V_u])

		# Right lane
        lower_right = np.array([self.right_H_l, self.right_S_l, self.right_V_l])
        upper_right = np.array([self.right_H_u, self.right_S_u, self.right_V_u])

        # Step line
        lower_stepline1 = np.array([self.stepline1_H_l, self.stepline1_S_l, self.stepline1_V_l])
        upper_stepline1 = np.array([self.stepline1_H_u, self.stepline1_S_u, self.stepline1_V_u])
        lower_stepline2 = np.array([self.stepline2_H_l, self.stepline2_S_l, self.stepline2_V_l])
        upper_stepline2 = np.array([self.stepline2_H_u, self.stepline2_S_u, self.stepline2_V_u])


		# Create masks for left and right colors
        mask_left = cv.inRange(hsv, lower_left, upper_left)
        mask_right = cv.inRange(hsv, lower_right, upper_right)
        mask_stepline1 = cv.inRange(hsv, lower_stepline1, upper_stepline1)
        mask_stepline2 = cv.inRange(hsv, lower_stepline2, upper_stepline2)
        masked_stepline_inter = cv.bitwise_or(mask_stepline1, mask_stepline2)


		# Bitwise AND masks with original image
        masked_left = cv.bitwise_and(image, rectangle, mask=mask_left)
        masked_right = cv.bitwise_and(image, rectangle, mask=mask_right)

		# Combine masked images
        masked_frame = masked_left + masked_right

        
        if not self.sim:
            rectangle[:350, : : ] = 0
        else:
            rectangle[:, :60, :] = 0
            rectangle[:, 260:, :] = 0

        masked_stepline = cv.bitwise_and(image, rectangle, mask=masked_stepline_inter)

        if (np.argwhere(masked_stepline > 0).any()):
            self.detect_stepline = True
        else:
            if self.detect_stepline:
                print("STEP + 1")
                self.step +=1
                self.detect_stepline = False


        self.left_lane = np.mean(np.argwhere(masked_left > 0),axis = 0)[:2]
        self.right_lane = np.mean(np.argwhere(masked_right > 0),axis = 0)[:2]

        center_left = tuple(np.intp(self.left_lane))  # Convert centroid to integer tuple
        center_left = tuple([center_left[1], center_left[0]])
        center_right = tuple(np.intp(self.right_lane))  # Convert centroid to integer tuple
        center_right = tuple([center_right[1], center_right[0]])

        # center_road = tuple([int(center_of_road), 210])


        radius = 10 
        color = (0, 255, 0)  # Green 
        color2 = (255, 255, 0) # Cyan
        thickness = 2 

        # Draw the circle on the self.image
        cv.circle(self.image, center_left, radius, color, thickness)
        cv.circle(self.image, center_right, radius, color2, thickness)
        # cv.circle(self.image, center_road, radius, color2, 5)

        image_message = self.bridge.cv2_to_imgmsg(rectangle, "passthrough")
        self.image_pub.publish(image_message)



if __name__ == "__main__":
    rospy.init_node("lane_detection", anonymous = True)
    camera_process = CameraProcess()
    rate = rospy.Rate(10)
    while(not rospy.is_shutdown()):
        camera_process.make_cmd()
        rate.sleep()
    stop = Twist()
    stop.angular.z = 0
    stop.linear.x = 0
    camera_process.cmd_vel_pub.publish(stop)