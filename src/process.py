#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2 as cv
from scipy import signal
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
            self.occupancy_grid_pub = rospy.Publisher('occupancy_grid_road', OccupancyGrid, queue_size=4)



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

            self.left_lane = [np.nan, np.nan]  
            self.right_lane = [np.nan, np.nan]


            self.LOOKAHEAD = 1 #meters
            self.CELLS_PER_METER = 50
            self.IS_FREE = 0
            self.IS_OCCUPIED = 100


            self.bridge = CvBridge()

            self.last_detection = None
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


        # print("ang_vel = ", self.ang_vel)
        self.cmd_twist.linear.x = self.cmd_speed
        self.cmd_twist.angular.z = self.ang_vel

        # print(self.cmd_twist)

        #speed pub
        # self.cmd_vel_pub.publish(self.cmd_twist)



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

        dir = min(10, max(-10, dir))

        print("left ", dst_left)
        print("right ", dst_right)

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

        if self.left_lane is None or self.right_lane is None:
            return
    
        rows, cols = self.image.shape[:2]
        
        speed = 0.2

        if self.step == 2:
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

        self.error = max(min(10, self.error),-10)

        print("error", self.error)

        # print("img shape", self.image.shape)

        print("center of road", center_of_road)

        try:
            self.error_pub.publish(Int32(int(self.error)))
        except:
            self.error_pub.publish(Int32(0))


        Kp = -0.1

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

        lower_yellow = np.array([self.left_H_l,self.left_S_l,self.left_V_l])
        upper_yellow = np.array([self.left_H_u,self.left_S_u,self.left_V_u])

        # Define HSV thresholds for red color
        lower_right = np.array([self.right_H_l, self.right_S_l, self.right_V_l])
        upper_right = np.array([self.right_H_u, self.right_S_u, self.right_V_u])



		# Create masks for yellow and white colors
        mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
        mask_right = cv.inRange(hsv, lower_right, upper_right)


        masked_frame = mask_yellow + mask_right

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

        self.merge_occup_grids()


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
        

        print("shp", self.occupancy_grid2.shape)

        lidar_occup = self.occupancy_grid

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


        # Convert image to HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        rectangle = np.copy(hsv)


        if self.sim:
            rectangle[:150, :, :] = 0
            # rectangle[220:, :, :] = 0
        else:
            rectangle[:, :40, :] = 0 
            rectangle[:, 280:, :] = 0



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
            rectangle[:350, : : ] = 0
        else:
            rectangle[:, :60, :] = 0
            rectangle[:, 280:, :] = 0

        mask_stepline1 = cv.inRange(rectangle, lower_stepline1, upper_stepline1)
        mask_stepline2 = cv.inRange(rectangle, lower_stepline2, upper_stepline2)
        mask_stepline_inter = cv.bitwise_or(mask_stepline1, mask_stepline2)

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
            masked_stepline = cv.bitwise_and(image, rectangle, mask=mask_stepline_inter)
        else:
            masked_stepline = cv.bitwise_and(image, rectangle, mask=mask_right)

        stepline_contour, _ = cv.findContours(mask_stepline_inter, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        if len(stepline_contour) > 0:
            largest_contour = max(stepline_contour, key=cv.contourArea)
            cv.drawContours(self.image, [largest_contour], -1, (0, 255, 0), 3)


        area = 0
        for i in stepline_contour:
            stepline_area = cv.contourArea(i)
            if stepline_area>area:
                area = stepline_area
        
        stepline_area = area


        # print("AREA STEPLINE  ", stepline_area)
        if (stepline_area > 2400 and stepline_area < 3200):
            
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


        contours_left, _ = cv.findContours(mask_left, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_right, _ = cv.findContours(mask_right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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

        image_message = self.bridge.cv2_to_imgmsg(self.image, "passthrough")
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