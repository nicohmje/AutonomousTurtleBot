#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import cv2 as cv
from scipy import signal
from sklearn.cluster import DBSCAN

from cv_bridge import CvBridge, CvBridgeError


from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import matplotlib.pyplot as plt
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry




import math


class Bottle:
    def __init__(self, index, position=None, gate=None, color=None):
        self.index = index
        self.position = position
        self.gate = gate
        self.color = color

    def get_position(self):
        return self.position

    def set_position(self, position:tuple):
        self.position = position

    def set_gate(self, gate:int):
        self.gate = gate

    def get_gate(self):
        return self.gate
    
    def get_index(self):
        return self.index
    
class Gate:
    def __init__(self, center_position=None, bottle_index1=None, bottle_index2=None, color=None):
        self.center_position = center_position
        self.bottle_index1 = bottle_index1
        self.bottle_index2 = bottle_index2

        self.confirmed = False
        self.color = color

        print("New gate alert! :", self.bottle_index1,self.bottle_index2)

    def confirm(self):
        self.confirmed = True

    def get_center_pos(self):
        return self.center_position
    

    def update(self, bottles):
        c1 = bottles[self.bottle_index1].get_position()
        c2 = bottles[self.bottle_index2].get_position()

        cX = int(np.mean([c1[0],c2[0]]))
        cY = int(np.mean([c1[1],c2[1]]))

        self.center_position= (cX, cY)
        



class CameraProcess:
    def __init__(self):
            self.image_pub = rospy.Publisher("/masked_frame", Image, queue_size=1)
            self.error_pub = rospy.Publisher("/error", Int32, queue_size=1)
            self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            self.occupancy_grid_pub = rospy.Publisher('occupancy_grid_road', OccupancyGrid, queue_size=4)
            


            rospy.Subscriber("/param_change_alert", Bool, self.get_params)

            self.laser_scan = None

            self.detect_stepline = False
            self.step = 5

            self.ang_vel = 0
            self.cmd_speed = 0


            self.min_angle_deg = -90
            self.max_angle_deg = 90

            self.occupancy_grid = None

            self.left_lane = [np.nan, np.nan]  
            self.right_lane = [np.nan, np.nan]


            self.LOOKAHEAD = 1 #meters
            self.CELLS_PER_METER = 50
            self.IS_FREE = 0
            self.IS_OCCUPIED = 100

            self.current_index = 0

            self.cv_image = None
            self.cv_image_rect = None


            self.bridge = CvBridge()

            self.last_detection = None
            self.get_params()



            self.bottles = []
            self.gates = []
            self.same_bottle_threshold = 3
            self.confirmed = False


        

            rospy.Subscriber("/camera/image", Image, self.callback_image)
            rospy.Subscriber("/occupancy_grid_noroad", OccupancyGrid, self.occupCB)
            rospy.Subscriber("/lidar_data", LaserScan, self.lidarCB)
            rospy.Subscriber("/image_rect_color", Image, self.callback_image_rect)
            rospy.Subscriber("/odom", Odometry, self.odomCB)


    def odomCB(self, msg:Odometry):
        pass



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

        print("Step:", self.step)
        if not self.step:
            self.cmd_speed = 0.2
            self.ang_vel = 0

        if self.step <= 1:
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

        elif self.step == 4:
            self.lane_assist_controller()

        elif self.step == 5:
            if self.cv_image_rect is None:
                print("NO IAMGE RECT")
            else:
                self.last_challenge()

        # print("ang_vel = ", self.ang_vel)
        self.cmd_twist.linear.x = self.cmd_speed

        if self.sim:
            self.ang_vel = min(0.9, max(-0.9, self.ang_vel))
        self.cmd_twist.angular.z = self.ang_vel

        # print(self.cmd_twist)

        # speed pub
        # self.cmd_vel_pub.publish(self.cmd_twist)


    def lidarCB(self, data:LaserScan) :
        self.laser_scan = data


    def lidar_only(self):
        # center_pos_index = int(len(self.laser_scan.ranges)//2 + 1)

        # # print("ang_vel", self.ang_vel)

        # offset = 75

        
        Kp = (4.0, -0.05)[self.sim]

        # print(90 - np.argmax(self.laser_scan.ranges))

        # dir = Kp * (90 - np.argmax(self.laser_scan.ranges))

        # print("dir ", dir)

        dst_left = np.mean(self.laser_scan.ranges[120:179])
        dst_right = np.mean(self.laser_scan.ranges[0:60])



        dir = Kp * (dst_left-dst_right)

        print(dir)

        dir = min(10, max(-10, dir))

        print("left ", dst_left)
        print("right ", dst_right)


        if abs(dir) > 0.1:
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


    def get_params(self, event=True):
        rospy.loginfo("Updating the parameters")

        try:
            self.sim = rospy.get_param("use_sim_time")
        except:
            self.sim = False

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




        #Blue Color Gains
        self.blue_H_l = rospy.get_param("/blue_H_l", default=77)
        self.blue_S_l = rospy.get_param("/blue_S_l", default=32)
        self.blue_V_l = rospy.get_param("/blue_V_l", default=76)

        self.blue_H_u = rospy.get_param("/blue_H_u", default=102)
        self.blue_S_u = rospy.get_param("/blue_S_u", default=180)
        self.blue_V_u = rospy.get_param("/blue_V_u", default=132)

        # Yellow Color Gains
        self.yellow_H_l = rospy.get_param("/yellow_H_l", default=100)
        self.yellow_S_l = rospy.get_param("/yellow_S_l", default=74)
        self.yellow_V_l = rospy.get_param("/yellow_V_l", default=123)

        self.yellow_H_u = rospy.get_param("/yellow_H_u", default=120)
        self.yellow_S_u = rospy.get_param("/yellow_S_u", default=255)
        self.yellow_V_u = rospy.get_param("/yellow_V_u", default=255)

        # Green Color Gains
        self.green_H_l = rospy.get_param("/green_H_l", default=100)
        self.green_S_l = rospy.get_param("/green_S_l", default=74)
        self.green_V_l = rospy.get_param("/green_V_l", default=123)

        self.green_H_u = rospy.get_param("/green_H_u", default=120)
        self.green_S_u = rospy.get_param("/green_S_u", default=255)
        self.green_V_u = rospy.get_param("/green_V_u", default=255)


    def callback_image(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_image(self.cv_image)
        

    def callback_image_rect(self, msg):
        self.cv_image_rect = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.second_process(self.cv_image_rect)


    def obstacle_detection(self):

        if self.occupancy_grid is None:
            return False
        # print("Obstacles left front right ", self.obs_left, self.obs_front, self.obs_right)
        
        print(self.occupancy_grid.shape)

        MARGIN = 4
        current_pos = (25,0)
        goal_pos = (25,10)


        error = 25 - (np.mean(self.check_collision(current_pos, goal_pos, margin=MARGIN), axis = 0))

        # print("error", error)

        if (np.isnan(error)).any():
            pass
        else:
            print("Obstacle!", error[0])
            self.ang_vel = error[0] * 2.0


    def lane_assist_controller(self):

        if self.left_lane is None or self.right_lane is None:
            return
    
        rows, cols = self.image.shape[:2]
        
        speed = 0.2

        if self.step == 2 and self.sim:
            speed = 0.1

        print("left, right", self.left_lane, self.right_lane)

        if self.left_lane[1]> self.image.shape[1]//2:
            self.left_lane[1] = 0
        if self.right_lane[1]< self.image.shape[1]//2:
            self.right_lane[1] = self.image.shape[1]

        print("left, right", self.left_lane, self.right_lane)


        if np.isnan(self.left_lane[1]):
            if np.isnan(self.right_lane[1]):
                print("left and right nan")
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

        print("width ", self.right_lane[1] - self.left_lane[1], "\n")

        max_width = (550, 240)[self.sim]
        offset = (0.45, 0.3)[self.sim]

        if (self.right_lane[1] - self.left_lane[1]) > max_width:
            print("Road split detected, following yellow line.")
            print("left lane, cols", self.left_lane[1], cols*offset)
            center_of_road = self.left_lane[1] + (cols * offset)


        self.error = center_of_road - self.image.shape[1]//2

        # self.error = max(min(10, self.error),-10)

        print("error", self.error)

        # print("img shape", self.image.shape)

        print("center of road", center_of_road)

        try:
            self.error_pub.publish(Int32(int(self.error)))
        except:
            self.error_pub.publish(Int32(0))

        if self.sim:
            Kp = -0.05 #-0.2 in si,m
        else:
            Kp = -0.02

        self.cmd_speed =  speed #0.22
        self.ang_vel = self.error * Kp#0.03

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


        if self.sim:
            lower_yellow = np.array([self.left_H_l,self.left_S_l,self.left_V_l])
            upper_yellow = np.array([self.left_H_u,self.left_S_u,self.left_V_u])

            # Define HSV thresholds for red color
            lower_right = np.array([self.right_H_l, self.right_S_l, self.right_V_l])
            upper_right = np.array([self.right_H_u, self.right_S_u, self.right_V_u])

            # Create masks for yellow and white colors
            mask_left = cv.inRange(hsv, lower_yellow, upper_yellow)
            mask_right = cv.inRange(hsv, lower_right, upper_right)

        else:
            lower_left = np.array([self.left_H_l, self.left_S_l, self.left_V_l])
            upper_left = np.array([self.left_H_u, self.left_S_u, self.left_V_u])

            lower_stepline1 = np.array([self.stepline1_H_l, self.stepline1_S_l, self.stepline1_V_l])
            upper_stepline1 = np.array([self.stepline1_H_u, self.stepline1_S_u, self.stepline1_V_u])
            lower_stepline2 = np.array([self.stepline2_H_l, self.stepline2_S_l, self.stepline2_V_l])
            upper_stepline2 = np.array([self.stepline2_H_u, self.stepline2_S_u, self.stepline2_V_u])


            mask_left = cv.inRange(hsv, lower_left, upper_left)

            mask_stepline1 = cv.inRange(hsv, lower_stepline1, upper_stepline1)
            mask_stepline2 = cv.inRange(hsv, lower_stepline2, upper_stepline2)
            mask_stepline_inter = cv.bitwise_or(mask_stepline1, mask_stepline2)
            
            kernel = np.ones((15,15), np.uint8)
            mask_right = cv.morphologyEx(mask_stepline_inter, cv.MORPH_CLOSE, kernel)




        mask_frame = mask_left + mask_right

        return mask_frame

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
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            return None, None

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
    def fillCurves(self, img_shape, pts_left, pts_right):
        pts = np.hstack((pts_left, pts_right))
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype='uint8')
        cv.fillPoly(img, np.int_([pts]), (255,0,255))
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
        # polygon = [(int(0), int(height)), (int(1), int(130)), (int(width-1), int(130)), (int(width), int(height))]
        # masked_img = self.regionOfInterest(processed_img, polygon)

        if self.sim:
            source_points = np.float32([[101,140], [width-101,140], [-200,height], [width+200,height]])
            destination_points = np.float32([[0,0], [800,0], [0,450],[800, 450]])
        else:
            source_points = np.float32([[0,0], [width,0], [-850,height], [width+850,height]])
            destination_points = np.float32([[0,0], [500,0], [0,600],[500, 600]])

        warped_img_size = (800, 450)
        warped_img = self.warp(processed_img, source_points, destination_points, warped_img_size)        
    
        kernel = np.ones((41,41), np.uint8)
        opening = cv.morphologyEx(warped_img, cv.MORPH_CLOSE, kernel)



        warped_img_shape = (warped_img.shape)

        left_fit, right_fit = self.fitCurve(opening)
        if left_fit is None or right_fit is None:
            return
        pts_left, pts_right = self.findPoints(warped_img_shape, left_fit, right_fit)
        fill_curves = self.fillCurves(warped_img_shape, pts_left, pts_right)

        unwarped_fill_curves = self.unwarp(fill_curves, source_points, destination_points, (width, height))
        window1 = cv.addWeighted(image, 1, unwarped_fill_curves, 1, 0)

        road = self.warp(window1, source_points, destination_points, warped_img_size)


        road_mask = cv.inRange(road, np.array([250,0,250]), np.array([255,20,255]))
        road = cv.bitwise_and(road, road, mask=road_mask)

        road = cv.GaussianBlur(road,(11,11),0)
        road = cv.resize(road, (40,22))


        last_row = road[-1,:,:]
        tiled = np.tile(last_row[np.newaxis, :, :], (6,1,1))
        road = np.vstack((road, tiled))

    
        road = cv.cvtColor(road, cv.COLOR_BGR2GRAY)
        _,road = cv.threshold(road,40,255,cv.THRESH_BINARY)

        kernel = np.ones(shape=[2, 2])
        self.occupancy_grid2 = signal.convolve2d(
            road.astype("int"), kernel.astype("int"), boundary="symm", mode="same"
        )
        self.occupancy_grid2 = np.clip(self.occupancy_grid2, 0, 50)

        # self.merge_occup_grids()


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
                # print("Road is turning left ", avg_radius)
            # self.error = (5000. - avg_radius)/5000.

        # self.cmd_speed =  0.22 #0.22
        # self.ang_vel = (self.crosstrack_error[0] * -0.01) + self.heading_error[0] * -0.01

        # print(self.ang_vel)
        # print("Heading error ", self.heading_error[0])
        # print("Crosstrack error ", self.crosstrack_error[0])


        
        pass


    def merge_occup_grids(self):
        

        # print("shp", self.occupancy_grid2.shape)
        # print("shp", self.occupancy_grid.shape)


        lidar_occup = np.copy(self.occupancy_grid)

        lidar_occup = np.transpose(np.rot90(lidar_occup, k=2, axes=(1,0)))

        if self.sim:
            lidar_occup = lidar_occup[7:, 2:-8]
        else:
            rospy.logwarn("IMPLEMENT THIS")

        self.occupancy_grid2 = np.bitwise_or(lidar_occup, self.occupancy_grid2)

        self.publish_occupancy_grid()


    def publish_occupancy_grid(self):
        """
        Publish populated occupancy grid to ros2 topic
        Args:
            scan_msg (LaserScan): message from lidar scan topic
        """
        oc = OccupancyGrid()
        oc.header.frame_id = "base_footprint"
        oc.header.stamp = rospy.Time.now()
        oc.info.origin.position.y -= (((self.occupancy_grid2.shape[1] / 2)) / self.CELLS_PER_METER) - 0.05
        oc.info.width = self.occupancy_grid2.shape[0]
        oc.info.height = self.occupancy_grid2.shape[1]
        oc.info.resolution = 1 / self.CELLS_PER_METER
        oc.data = np.fliplr(np.rot90(self.occupancy_grid2, k=1)).flatten().tolist()
        self.occupancy_grid_pub.publish(oc)
        

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

        if self.step <= 4:
            # Convert image to HSV color space
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

            rectangle = np.copy(hsv)


            if self.sim:
                rectangle[:150, :, :] = 0
                # rectangle[220:, :, :] = 0
            else:
                pass
                # rectangle[]
                # rectangle[:, :40, :] = 0 
                # rectangle[:, 280:, :] = 0



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


            # Create masks for left, right and stepline colors
            mask_left = cv.inRange(rectangle, lower_left, upper_left)
            mask_right = cv.inRange(rectangle, lower_right, upper_right)



            kernel = np.ones((15,15), np.uint8)
            mask_left = cv.morphologyEx(mask_left, cv.MORPH_CLOSE, kernel)
            mask_right = cv.morphologyEx(mask_right, cv.MORPH_CLOSE, kernel)

            if not self.sim:
                pass
                # rectangle[:350, : : ] = 0
            else:
                pass
                # rectangle[:, :60, :] = 0
                # rectangle[:, 280:, :] = 0

            mask_stepline1 = cv.inRange(rectangle, lower_stepline1, upper_stepline1)
            mask_stepline2 = cv.inRange(rectangle, lower_stepline2, upper_stepline2)
            mask_stepline_inter = cv.bitwise_or(mask_stepline1, mask_stepline2)

            mask_stepline_inter = cv.morphologyEx(mask_stepline_inter, cv.MORPH_CLOSE, kernel)



            # In the real world, the right lane is red, so we use the red color (with the two ranges). 
            # It's easier to just switch stepline and right than to change everything. 

            if self.sim:
                masked_left = cv.bitwise_and(image, rectangle, mask=mask_left)
                masked_right = cv.bitwise_and(image, rectangle, mask=mask_right)
            else:
                masked_left = cv.bitwise_and(image, rectangle, mask=mask_left)
                masked_right = cv.bitwise_and(image, rectangle, mask=mask_stepline_inter)



            
            # Combine masked images
        
            masked_frame = masked_left + masked_right



            if self.sim:
                stepline_contour, _ = cv.findContours(mask_stepline_inter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                masked_stepline = cv.bitwise_and(image, rectangle, mask=mask_stepline_inter)
            else:
                stepline_contour, _ = cv.findContours(mask_right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                masked_stepline = cv.bitwise_and(image, rectangle, mask=mask_right)

            
            if len(stepline_contour) > 0:
                largest_contour = max(stepline_contour, key=cv.contourArea)
                cv.drawContours(self.image, [largest_contour], -1, (0, 255, 0), 3)


            area = 0
            for i in stepline_contour:
                stepline_area = cv.contourArea(i)
                if stepline_area>area:
                    area = stepline_area
            
            stepline_area = area


            print("AREA STEPLINE  ", stepline_area)

            lower = (15000,2400)[self.sim]
            upper = (30000,3200)[self.sim]


            if (stepline_area > lower and stepline_area < upper):
                
                if (not self.last_detection is None) and (((rospy.Time.now() - self.last_detection).to_sec()) < 10):
                    print("folse detekshion", self.last_detection.to_sec(), ((rospy.Time.now() - self.last_detection).to_sec()))
                    pass
                else:
                    self.detect_stepline = True
            else:
                if self.detect_stepline:
                    print("STEP + 1, AREA: ", stepline_area)
                    self.step +=1
                    self.last_detection = rospy.Time.now()
                    self.detect_stepline = False


            if self.sim:
                contours_left, _ = cv.findContours(mask_left, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours_right, _ = cv.findContours(mask_right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            else:
                contours_left, _ = cv.findContours(mask_left, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours_right, _ = cv.findContours(mask_stepline_inter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            left_lane, right_lane = None,None
            # Find the largest contour based on area
            if len(contours_left)>0:
                left_lane = max(contours_left, key=cv.contourArea)
                cv.drawContours(self.image, [left_lane], -1, (0, 255, 0), 3)
            if len(contours_right)>0:
                right_lane = max(contours_right, key=cv.contourArea)
                cv.drawContours(self.image, [right_lane], -1, (255, 255, 0), 3)


            # Calculate the centroid of the largest contour

            if not self.left_lane is None:
                M = cv.moments(left_lane)
            if M["m00"] != 0:
                self.left_lane[1] = int(M["m10"] / M["m00"])
                self.left_lane[0] = int(M["m01"] / M["m00"])
            else:
                self.left_lane = [np.nan, np.nan]
            # Calculate the centroid of the largest contour
            if not self.right_lane is None:
                M = cv.moments(right_lane)
            if M["m00"] != 0:
                self.right_lane[1] = int(M["m10"] / M["m00"])
                self.right_lane[0] = int(M["m01"] / M["m00"])
            else:
                self.right_lane = [np.nan, np.nan]


            center_left = tuple([self.left_lane[1], self.left_lane[0]])
            center_right = tuple([self.right_lane[1], self.right_lane[0]])

            radius = 10 
            color = (0, 255, 0)  # Green 
            color2 = (255, 255, 0) # Cyan
            thickness = 2 

            # Draw the circle on the self.image
            if not np.isnan(center_left[0]):
                cv.circle(self.image, center_left, radius, color, thickness)
            if not np.isnan(center_right[0]):
                cv.circle(self.image, center_right, radius, color2, thickness)
            # cv.circle(self.image, center_road, radius, color2, 5)


            image_message = self.bridge.cv2_to_imgmsg(mask_right, "passthrough")
            self.image_pub.publish(image_message)
        else:
            pass

    def last_challenge(self):
        if self.occupancy_grid is None:
            return
        

        lidar_occup = np.copy(self.occupancy_grid.filled(0))
        lidar_occup = np.transpose(np.rot90(lidar_occup, k=2, axes=(1,0)))


        self.occupancy_grid2 = np.zeros((lidar_occup.shape[0], lidar_occup.shape[1], 3), dtype=np.uint8)

        self.occupancy_grid2 = cv.cvtColor(self.occupancy_grid2, cv.COLOR_BGR2GRAY)

        OccuGrid2 = (np.argwhere(lidar_occup == 100))

        clustering = DBSCAN(eps=3, min_samples=10).fit(OccuGrid2)
        
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("clusters ", n_clusters_)


        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = OccuGrid2[class_member_mask & core_samples_mask]
            if len(xy)>0:
                x = int(np.mean(xy[:,0]))
                y = int(np.mean(xy[:,1]))
                matched = False
                for i in self.bottles:
                    x2, y2 = i.get_position()
                    distance = math.sqrt((x2 - x)**2 + (y2 - y)**2)
                    if abs(distance) < self.same_bottle_threshold:
                        # print(f"matched with an existing bottle, distance: {distance}")
                        i.set_position((x,y))
                        if i.get_gate():
                            self.gates[i.get_gate()].update(self.bottles)
                        matched = True
                        break
                    else:
                        continue
                if not matched:
                    print("new bottle discovered!")
                    self.bottles.append(Bottle(len(self.bottles), position=(x,y)))
                # cv.circle(self.occupancy_grid2, [x,y], 4, (0,255,255), 4)


        potential_neighbors = {}
        neighbors = []
        res_neighbors = {}

        target_distance = 26.5
        tolerance = 3
        if not self.confirmed:
            for i in range(len(self.bottles)):
                potential_neighbors[i] = []
                for j in range(len(self.bottles)):
                        if i==j:
                            continue
                        x1, y1 = self.bottles[i].get_position()
                        x2, y2 = self.bottles[j].get_position()
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if abs(distance - target_distance) < tolerance:
                            potential_neighbors[i].append(j)
                            # print(distance)
                            # print(i,j)
                            
            print("potential", potential_neighbors)

            for i in potential_neighbors:
                if len(potential_neighbors[i]) == 1:
                    if len(potential_neighbors[potential_neighbors[i][0]]) == 1:
                        if potential_neighbors[potential_neighbors[i][0]][0] == i:
                            # print(f"Neighbors found: {i}, {potential_neighbors[i]}")
                            if not any(i in gate for gate in neighbors):
                                neighbors.append((i, potential_neighbors[i][0]))
                    else: 
                        try:
                            res_neighbors[potential_neighbors[i][0]].append(i)
                        except:
                            res_neighbors[potential_neighbors[i][0]] = [i]

            for i in res_neighbors:
                if len(res_neighbors[i])==1:
                    neighbors.append((i, res_neighbors[i][0]))
                else:
                    print(f"ambiguity for {i}")

            print("neighbors:", neighbors)


            for i in neighbors:

                print((self.bottles[i[0]].get_gate(), (self.bottles[i[1]].get_gate())))
                if self.bottles[i[0]].get_gate() is None or self.bottles[i[1]].get_gate() is None:
                    pass
                else:
                    if self.bottles[i[0]].get_gate() == self.bottles[i[1]].get_gate():
                        continue
                c1 = self.bottles[i[0]].get_position()
                c2 = self.bottles[i[1]].get_position()

                cX = int(np.mean([c1[0],c2[0]]))
                cY = int(np.mean([c1[1],c2[1]]))

                # self.occupancy_grid2[cX,cY] = 127

                self.bottles[i[0]].set_gate(len(self.gates))
                self.bottles[i[1]].set_gate(len(self.gates))
                
                self.gates.append(Gate(center_position=(cX,cY), bottle_index1=self.bottles[i[0]].get_index(), bottle_index2=self.bottles[i[1]].get_index()))
                # cv.circle(self.occupancy_grid2, [cX,cY], 2, (0,255,0), 3)

        if n_clusters_ == 6 and len(neighbors) == 3 and not self.confirmed:
            self.confirmed = True
            print("found all gates!")
            for i in self.gates:
                i.confirm()

        for i in self.bottles:
            # print(i.get_gate())
            self.occupancy_grid2[i.get_position()[0],i.get_position()[1]] = 100
            # cv.circle(occu_grid35, [i.get_position()[0],i.get_position()[1]], 1, (0,255,0), 2)  

        for i in self.gates:
            self.occupancy_grid2[i.get_center_pos()[0], i.get_center_pos()[1]] = 127
            # cv.circle(occu_grid35, [i.get_center_pos()[0], i.get_center_pos()[1]], 2,(0,0,255), 1)

        print("nb bottles: ", len(self.bottles))
        print("nb gates: ", len(self.gates))

        self.publish_occupancy_grid()
        pass

    def last_challenge2(self):
        self.image_rect = np.copy(self.cv_image_rect)

        hsv = cv.cvtColor(self.cv_image_rect, cv.COLOR_BGR2HSV)

        # Blue lane
        BlueBottle_l = np.array([self.blue_H_l, self.blue_S_l, self.blue_V_l])
        BlueBottle_u = np.array([self.blue_H_u, self.blue_S_u, self.blue_V_u])

		# Green lane
        GreenBottle_l = np.array([self.green_H_l, self.green_S_l, self.green_V_l])
        GreenBottle_u = np.array([self.green_H_u, self.green_S_u, self.green_V_u])

        # Yellow lane
        yellowBottle_l = np.array([self.yellow_H_l, self.yellow_S_l, self.yellow_V_l])
        yellowBottle_u = np.array([self.yellow_H_u, self.yellow_S_u, self.yellow_V_u])

        mask_blue = cv.inRange(hsv, BlueBottle_l, BlueBottle_u)
        mask_green = cv.inRange(hsv, GreenBottle_l, GreenBottle_u)
        mask_yellow = cv.inRange(hsv, yellowBottle_l, yellowBottle_u)

        contours_blue, _ = cv.findContours(mask_blue, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv.findContours(mask_green, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv.findContours(mask_yellow, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        threshold = 5000


        self.order = [0,1,2] #0 blue, 1 green, 2 yellow

        yellow_bottles = []
        green_bottles = []
        blue_bottles = []

        yellow_middle = None
        blue_middle = None
        green_middle = None

        blue_x = 0
        yellow_x = 0
        green_x = 0

        for i in contours_blue:
            if cv.contourArea(i) > threshold:
                print("contour blue area: ", cv.contourArea(i))
                M = cv.moments(i)
                if M["m00"] != 0:
                    y = int(M["m10"] / M["m00"])
                    x = int(M["m01"] / M["m00"])
                    blue_bottles.append([x,y])
            cv.drawContours(self.image_rect, [i], -1, (255, 0, 0), 3)



        for i in contours_green:
            if cv.contourArea(i) > threshold:
                print("contour green area: ", cv.contourArea(i))
                M = cv.moments(i)
                if M["m00"] != 0:
                    y = int(M["m10"] / M["m00"])
                    x = int(M["m01"] / M["m00"])
                    green_bottles.append([x,y])
            cv.drawContours(self.image_rect, [i], -1, (0, 255, 0), 3)

        for i in contours_yellow:
            if cv.contourArea(i) > threshold:
                
                print("contour yellow area: ", cv.contourArea(i))
                M = cv.moments(i)
                if M["m00"] != 0:
                    y = int(M["m10"] / M["m00"])
                    x = int(M["m01"] / M["m00"])
                    yellow_bottles.append([x,y])
            cv.drawContours(self.image_rect, [i], -1, (0, 0, 255), 3)

        self.cmd_speed = 0.05

        print('len yellow', len(yellow_bottles))
        if len(yellow_bottles) > 0:
            if len(yellow_bottles) == 1:
                if yellow_bottles[0][1] > 320:
                    yellow_middle = 640
                else:
                    yellow_middle = 0
            
            elif len(yellow_bottles) == 2:
                yellow_middle = int(abs(yellow_bottles[1][1] + 0.5*(yellow_bottles[0][1] - yellow_bottles[1][1])))
                yellow_x = abs(yellow_bottles[1][0] + 0.5*(yellow_bottles[0][0] - yellow_bottles[1][0]))
            print("yellow middle :", yellow_middle)
            print("x, :", yellow_x)


        print('len green', len(green_bottles))

        if len(green_bottles) > 0:
            if len(green_bottles) == 1:
                if green_bottles[0][1] > 320:
                    green_middle = 640
                else:
                    green_middle = 0

            elif len(green_bottles) == 2:
                green_middle = int(abs(green_bottles[1][1] + 0.5*(green_bottles[0][1] - green_bottles[1][1])))
                green_x = abs(green_bottles[1][0] + 0.5*(green_bottles[0][0] - green_bottles[1][0]))
            print("green middle :", green_middle)
            print("yellow x, :", green_x)
        
        print('len blue', len(blue_bottles))

        if len(blue_bottles) > 0:
            if len(blue_bottles) == 1:
                if blue_bottles[0][1] > 320:
                    blue_middle = 640
                else:
                    blue_middle = 0
            elif len(blue_bottles) == 2: 
                blue_middle = int(abs(blue_bottles[1][1] + 0.5*(blue_bottles[0][1] - blue_bottles[1][1])))
                blue_x = abs(blue_bottles[1][0] + 0.5*(blue_bottles[0][0] - blue_bottles[1][0]))
            print("blue middle :", blue_middle)
            print("blue x:", blue_x)


        if yellow_middle is None and blue_middle is None and green_middle is None:
            self.ang_vel = -1.0
            self.cmd_speed = 0.1



        radius = 10 
        color = (0, 255, 0)  # Green 
        color3 = (255,0,0) #Blue
        color2 = (0, 0, 255) #Red
        thickness = 2 

        

        if not yellow_middle is None:
            cv.circle(self.image_rect, [yellow_middle, yellow_bottles[0][0]], radius, color2, thickness)
            yellow_error = yellow_middle - (self.image_rect.shape[1] *0.5)
            print(self.image_rect.shape)
            print(yellow_error)

        if not blue_middle is None:
            cv.circle(self.image_rect, [blue_middle, blue_bottles[0][0]], radius, color3, thickness)
            blue_error = blue_middle - (self.image_rect.shape[1] *0.5)
            print(self.image_rect.shape)
            print(blue_error)  

        if not green_middle is None:
            cv.circle(self.image_rect, [green_middle, green_bottles[0][0]], radius, color, thickness)
            green_error = green_middle - (self.image_rect.shape[1] *0.5)
            print(self.image_rect.shape)
            print(green_error)  

        #1 blue, 2 green, 3 yellow

        print("current target", self.current_index, self.order[self.current_index])
        if self.order[self.current_index] == 0:
            if blue_x > 300:
                self.current_index += 1
            print("ORDER BLUE")
            if blue_middle is None:
                if not green_middle is None:
                    print("1")
                    self.cmd_speed = 0
                    self.ang_vel = green_error * 0.05
                if not yellow_middle is None:
                    print("2")
                    self.cmd_speed = 0
                    self.ang_vel = yellow_error * 0.05
            else:
                print("3")
                self.ang_vel = blue_error * -0.005
        
        elif self.order[self.current_index] == 1:
            if green_x > 300:
                self.current_index += 1
            print("ORDER GREEN")
            if green_middle is None:
                if not blue_middle is None:
                    print("4")
                    self.cmd_speed = 0
                    self.ang_vel = blue_error * 0.05
                if not yellow_middle is None:
                    print("5")
                    self.cmd_speed = 0
                    self.ang_vel = yellow_error * 0.05
            else:
                print("6")
                self.ang_vel = green_error * -0.005

        elif self.order[self.current_index] == 2:
            if len(yellow_bottles) > 0:
                print("I see a yellow bottle !")
        
            if yellow_x > 300:
                self.current_index += 1
            print("ORDER YELLOW")
            if yellow_middle is None:
                if not blue_middle is None:
                    print("7")
                    self.cmd_speed = 0
                    self.ang_vel = blue_error * 0.05
                elif not green_middle is None:
                    print("8")
                    self.cmd_speed = 0
                    self.ang_vel = green_error * 0.05
            else:
                print("9")
                self.ang_vel = yellow_error * -0.005
        
        
        if not self.occupancy_grid is None:
            np.save("occupgrid", self.occupancy_grid)



        image_message = self.bridge.cv2_to_imgmsg(self.image_rect, "passthrough")
        self.image_pub.publish(image_message)

        pass


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