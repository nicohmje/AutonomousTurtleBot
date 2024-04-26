"""

Path planning with Rapidly-Exploring Random Trees (RRT)

author: Aakash(@nimrobotics)
web: nimrobotics.github.io

"""

import cv2
import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid
import random

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []


class RRT:


    def __init__(self):
        rospy.init_node("RRT_obstacles")


        self.node_list = [0]
        self.stepSize = 2

        rospy.Subscriber("occupancy_grid_noroad", OccupancyGrid, self.occup_grid1_CB)
        rospy.Subscriber("occupancy_grid_road", OccupancyGrid, self.occup_grid2_CB)

        rospy.spin()


    def occup_grid1_CB(self, msg):
        data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.occupancy_grid1 = np.ma.array(data, mask=data==-1, fill_value=-1)
        pass

    def occup_grid2_CB(self, msg):
        data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.occupancy_grid2 = np.ma.array(data, mask=data==-1, fill_value=-1)
        pass

    def combineOccupGrid(self):
        


    # check collision
    def collision(self, x1,y1,x2,y2):
        color=[]
        x = list(np.arange(x1,x2,(x2-x1)/100))
        y = list(((y2-y1)/(x2-x1))*(x-x1) + y1)
        print("collision",x,y)
        for i in range(len(x)):
            print(int(x[i]),int(y[i]))
            color.append(self.occup_grid[int(y[i]),int(x[i])])
        if (255 in color):
            return True #collision
        else:
            return False #no-collision

    # check the  collision with obstacle and trim
    def check_collision(self, x1,y1,x2,y2):
        _,theta = self.dist_and_angle(x2,y2,x1,y1)
        x=x2 + self.stepSize*np.cos(theta)
        y=y2 + self.stepSize*np.sin(theta)
        print(x2,y2,x1,y1)
        print("theta",theta)
        print("check_collision",x,y)

        # TODO: trim the branch if its going out of image area
        # print("Image shape",self.occup_grid.shape)
        hy,hx=self.occup_grid.shape
        if y<0 or y>hy or x<0 or x>hx:
            print("Point out of image bound")
            directCon = False
            nodeCon = False
        else:
            # check direct connection
            if self.collision(x,y,self.end[0],self.end[1]):
                directCon = False
            else:
                directCon=True

            # check connection between two nodes
            if self.collision(x,y,x2,y2):
                nodeCon = False
            else:
                nodeCon = True

        return(x,y,directCon,nodeCon)

    # return dist and angle b/w new point and nearest node
    def dist_and_angle(self, x1,y1,x2,y2):
        dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
        angle = math.atan2(y2-y1, x2-x1)
        return(dist,angle)

    # return the neaerst node index
    def nearest_node(self, x,y):
        temp_dist=[]
        for i in range(len(self.node_list)):
            dist,_ = self.dist_and_angle(x,y,self.node_list[i].x,self.node_list[i].y)
            temp_dist.append(dist)
        return temp_dist.index(min(temp_dist))

    # generate a random point in the image space
    def rnd_point(self, h,l):
        new_y = random.randint(0, h)
        new_x = random.randint(0, l)
        return (new_x,new_y)


    def FindPath(self, g2, start, end):
        h,l= self.occup_grid.shape # dim of the loaded image
        
        # insert the starting point in the node class
        # self.node_list = [0] # list to store all the node points         
        self.node_list[0] = Nodes(start[0],start[1])
        self.node_list[0].parent_x.append(start[0])
        self.node_list[0].parent_y.append(start[1])

        i=1
        pathFound = False
        while pathFound==False:
            nx,ny = self.rnd_point(h,l)
            print("Random points:",nx,ny)

            nearest_ind = self.nearest_node(nx,ny)
            nearest_x = self.node_list[nearest_ind].x
            nearest_y = self.node_list[nearest_ind].y
            print("Nearest node coordinates:",nearest_x,nearest_y)

            #check direct connection
            tx,ty,directCon,nodeCon = self.check_collision(nx,ny,nearest_x,nearest_y)
            print("Check collision:",tx,ty,directCon,nodeCon)

            if directCon and nodeCon:
                print("Node can connect directly with end")
                self.node_list.append(i)
                self.node_list[i] = Nodes(tx,ty)
                self.node_list[i].parent_x = self.node_list[nearest_ind].parent_x.copy()
                self.node_list[i].parent_y = self.node_list[nearest_ind].parent_y.copy()
                self.node_list[i].parent_x.append(tx)
                self.node_list[i].parent_y.append(ty)

                print("Path has been found")
                break

            elif nodeCon:
                print("Nodes connected")
                self.node_list.append(i)
                self.node_list[i] = Nodes(tx,ty)
                self.node_list[i].parent_x = self.node_list[nearest_ind].parent_x.copy()
                self.node_list[i].parent_y = self.node_list[nearest_ind].parent_y.copy()
                # print(i)
                # print(self.node_list[nearest_ind].parent_y)
                self.node_list[i].parent_x.append(tx)
                self.node_list[i].parent_y.append(ty)
                i=i+1
                continue

            else:
                print("No direct con. and no node con. :( Generating new rnd numbers")
                continue

if __name__ == '__main__':
    # run the RRT algorithm 
    try:
        RRT()
    except:
        exit(0)
