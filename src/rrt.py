"""

Path planning with Rapidly-Exploring Random Trees (RRT)

author: Aakash(@nimrobotics)
web: nimrobotics.github.io

"""

import cv2
import numpy as np
import math
import random
import time
import argparse
from skimage.draw import line
import os

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.cost = float('inf')
        self.parent = None
        self.children = []


# check collision
def collision(x1, y1, x2, y2):
    # Simplified collision checking

    start = (int(x1), int(y1))
    end = (int(x2), int(y2))
    discrete_line = list(zip(*line(*start, *end)))
    for point in discrete_line:
        if img[point[1], point[0]] == 254:  # Assuming obstacle is white
            return True
    return False

# check the  collision with obstacle and trim
def check_collision(x1,y1,x2,y2):
    hy,hx=img.shape
    if y1<0 or y1>hy or x1<0 or x1>hx:
        # print("Point out of image bound")
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(x1,y1,end[0],end[1]):
            directCon = False
        else:
            dst, _ = dist_and_angle(x1,y1,end[0],end[1])
            if dst > stepSize*2:
                directCon = False
            else:
                directCon=True

        # check connection between two nodes
        if collision(x1,y1,x2,y2):
            nodeCon = False
        else:
            nodeCon = True

    return(directCon,nodeCon)

# return dist and angle b/w new point and nearest node
def dist_and_angle(x1,y1,x2,y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2-y1, x2-x1)
    return(dist,angle)

def distance(n1, n2):
    dist = math.sqrt( ((n2.x-n1.x)**2)+((n2.y-n1.y)**2) )
    return dist


def near_nodes(new_node):
    return [node for node in node_list if distance(node, new_node) < radius]


def best_parent(nx, ny):
    best_cost = float('inf')
    best_node = None
    next_node = Nodes(nx, ny)
    for i in range(len(node_list)):
        n = node_list[i]
        dst = distance(n, next_node)
        if dst > radius:
            continue
        cost = n.cost + dst
        # print("cost, best", cost, best_cost, dst)
        if cost<best_cost and not collision(n.x, n.y, next_node.x, next_node.y):
            best_cost = cost
            best_node = i
    return best_node, best_cost

def rewire(new_node, near_nodes):
    for node in near_nodes:
        cost_via_new_node = new_node.cost + distance(new_node, node)
        # print("new, old cost", cost_via_new_node, node.cost)
        if cost_via_new_node < node.cost and not collision(new_node.x , new_node.y, node.x, node.y):
            if node.parent:
                node.parent.children.remove(node)
            node.parent = new_node
            new_node.children.append(node)
            node.cost = cost_via_new_node
            update_children_costs(node)  # Recursively update costs of all children

def update_children_costs(node):
    for child in node.children:
        proposed_cost = node.cost + distance(node, child)
        if proposed_cost < child.cost:
            # print("UPDATED CHILD NODE COST")
            child.cost = proposed_cost
            update_children_costs(child)  # Recursively update children's costs

# return the neaerst node index
def nearest_node(x,y):
    temp_dist=[]
    for i in range(len(node_list)):
        dist,_ = dist_and_angle(x,y,node_list[i].x,node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))

# generate a random point in the image space
def rnd_point(h,l):
    new_y = random.randint(0, h)
    new_x = random.randint(0, l)
    return (new_x,new_y)


def RRT(img, img2, start, end, stepSize):
    h,l= img.shape # dim of the loaded image
    # print(img.shape) # (384, 683)
    # print(h,l)

    # insert the starting point in the node class
    # node_list = [0] # list to store all the node points         
    node_list[0] = Nodes(start[0],start[1])
    node_list[0].parent = None
    node_list[0].cost = 0

    # display start and end
    cv2.circle(img2, (start[0],start[1]), 1,(0,0,255),thickness=1, lineType=8)
    cv2.circle(img2, (end[0],end[1]), 1,(0,0,255),thickness=1, lineType=8)

    i=1
    pathFound = False
    while pathFound==False:
        # time.sleep(3)
        nx,ny = rnd_point(h,l)
        # print("Random points:",nx,ny)

        # cv2.circle(img2, (int(nx),int(ny)), 4,(0,255,255),thickness=4, lineType=8)

        nearest_ind = nearest_node(nx,ny)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y

        # print("Nearest", nearest_ind)

        _,theta = dist_and_angle(nearest_x,nearest_y,nx,ny)
        nx=int(nearest_x + stepSize*np.cos(theta))
        ny=int(nearest_y + stepSize*np.sin(theta))

        # cv2.circle(img2, (int(nx),int(ny)), 4,(255,255,0),thickness=4, lineType=8)

        hy,hx=img.shape
        if ny<0 or ny>hy or nx<0 or nx>hx:
            continue

        nearest_ind, new_cost = best_parent(nx,ny)

        # print("best", nearest_ind, new_cost)

        if nearest_ind is None:
            # print("no closest")
            continue
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y
        # print("best node coordinates:",nearest_x,nearest_y)

        #check direct connection
        # print("nxny", nx, ny)
        tx, ty = nx, ny
        directCon,nodeCon = check_collision(tx,ty,nearest_x,nearest_y)
        # print("Check collision:",tx,ty,directCon,nodeCon)

        if directCon and nodeCon:
            # print("Node can connect directly with end")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            # node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            # node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            # node_list[i].parent_x.append(tx)
            # node_list[i].parent_y.append(ty)
            node_list[i].parent = node_list[nearest_ind]
            node_list[nearest_ind].children.append(node_list[i])
            node_list[i].cost = new_cost

            # cv2.circle(img2, (int(tx),int(ty)), 1,(0,0,255),thickness=1, lineType=8)
            # cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
            # cv2.line(img2, (int(tx),int(ty)), (end[0],end[1]), (255,0,0), thickness=1, lineType=8)

            print(f"Path has been found in {i} iterations")


            # print("parent_x",node_list[i].parent_x)
            node = node_list[i]
            pos = (node.x, node.y)
            checkpoint = []
            while True:
                old_pos = pos
                pos, node = get_parent_coord(node)
                checkpoint.append(pos)
                if node is None:
                    print("reached start")
                    break
                cv2.line(img2, (int(old_pos[0]),int(old_pos[1])), (int(pos[0]),int(pos[1])), (255,0,0), thickness=2, lineType=8)

        

            # for j in range(len(node_list[i].parent_x)-1):
            #     cv2.line(img2, (int(node_list[i].parent_x[j]),int(node_list[i].parent_y[j])), (int(node_list[i].parent_x[j+1]),int(node_list[i].parent_y[j+1])), (255,0,0), thickness=2, lineType=8)
            # cv2.waitKey(1)
            cv2.imwrite("media/"+str(i)+".jpg",img2)
            cv2.imwrite("out.jpg",img2)

            return checkpoint
            break

        elif nodeCon:
            # print("Nodes connected")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            # node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            # node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent = node_list[nearest_ind]
            node_list[nearest_ind].children.append(node_list[i])
            # print(i)
            # print(node_list[nearest_ind].parent_y)
            # node_list[i].parent_x.append(tx)
            # node_list[i].parent_y.append(ty)
            node_list[i].cost = new_cost

            

            rewire(node_list[i], near_nodes(node_list[i]))
            i=i+1
            # display
            # cv2.circle(img2, (int(tx),int(ty)), 1,(0,0,255),thickness=3, lineType=8)
            # cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
            # # cv2.imwrite("media/"+str(i)+".jpg",img2)
            # cv2.imshow("sdc",img2)
            # cv2.waitKey(1)
            continue

        else:
            # print("No direct con. and no node con. :( Generating new rnd numbers")
            continue

def draw_circle(event,x,y,flags,param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2,(x,y),5,(255,0,0),-1)
        coordinates.append(x)
        coordinates.append(y)

def get_parent_coord(node):
    if node.parent is None:
        return start, None
    x = node.parent.x 
    y = node.parent.y
    parent = node.parent

    return (x,y), parent



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-p', type=str, default='world2.png',metavar='ImagePath', action='store', dest='imagePath',
                    help='Path of the image containing mazes')
    parser.add_argument('-s', type=int, default=4,metavar='Stepsize', action='store', dest='stepSize',
                    help='Step-size to be used for RRT branches')
    parser.add_argument('-start', type=int, default=[20,20], metavar='startCoord', dest='start', nargs='+',
                    help='Starting position in the maze')
    parser.add_argument('-stop', type=int, default=[450,250], metavar='stopCoord', dest='stop', nargs='+',
                    help='End position in the maze')
    parser.add_argument('-selectPoint', help='Select start and end points from figure', action='store_true')

    args = parser.parse_args()

    # remove previously stored data
    try:
      os.system("rm -rf media")
    except:
      print("Dir already clean")
    os.mkdir("media")

    img = cv2.imread(args.imagePath,0) # load grayscale maze image
    img2 = cv2.imread(args.imagePath) # load colored maze image
    start = tuple(args.start) #(20,20) # starting coordinate
    end = tuple(args.stop) #(450,250) # target coordinate
    stepSize = args.stepSize # stepsize for RRT
    node_list = [0] # list to store all the node points

    gamma = 2**2 * (1 + 1/2)
    
    radius = 20

    coordinates=[]
    if args.selectPoint:
        print("Select start and end points by double clicking, press 'escape' to exit")
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',draw_circle)
        while(1):
            cv2.imshow('image',img2)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        # print(coordinates)
        start=(coordinates[0],coordinates[1])
        end=(coordinates[2],coordinates[3])

    # run the RRT algorithm 
    print(RRT(img, img2, start, end, stepSize))