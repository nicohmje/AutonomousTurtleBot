import math
import random
import numpy as np
import rospy
from skimage.draw import line


"""
    Classes to handle RRT* path planning, used for the bottle challenge.
    It uses an occupancy grid to plan around the obstacles.
"""



class Nodes:
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.cost = float('inf')
        self.parent = None
        self.children = []

class RRTStarPlanning:
    def __init__(self, stepSize=4, radius=20, max_iters=600, cpm=50, is_occupied=0):
        self.occugrid = None
        self.radius = radius
        self.stepSize= stepSize
        self.max_iters = max_iters
        self.cpm = cpm
        self.occup = is_occupied
        self.busy = False

    def set_occugrid(self, occugrid):
        
        """
            Save the occu grid which contains the obstacles.
        """
        
        if self.busy:
            return
        self.occugrid = occugrid

        #The turtlebot is at this position, as this is how the occugrid is made.
        self.start = (self.occugrid.shape[0]-1, occugrid.shape[1]//2)
    

    def plan(self, end):
        if self.occugrid is None:
            return [None]
        self.node_list = [0]
        self.end = end
        return self.RRT()
    
    def rel_to_grid(self, rel):
        """ transform from relative to occugrid """
        x = (self.occugrid.shape[0]-1) - (rel[0]*self.cpm)
        y = (self.occugrid.shape[1]//2) - (rel[1]*self.cpm)
        return (int(x),int(y))
    
    def grid_to_rel(self, grid):
        x = (self.occugrid.shape[0]-1 - grid[0])/self.cpm
        y = ((self.occugrid.shape[1]//2) - grid[1])/self.cpm
        return (x, y)
    
    def collision(self, x1, y1, x2, y2):
        # Switched from bresenham to skimage.draw.line
        discrete_line = (zip(*line(x1,y1,x2,y2)))
        for point in discrete_line:
            try:
                if self.occugrid[point[0], point[1]] == self.occup:  # Assuming obstacle is white
                    return True
            except:
                continue
        return False

    def check_collision(self, x1,y1,x2,y2):

        """ Check path between two nodes. """

        hx,hy=self.occugrid.shape
        if y1<0 or y1>hy or x1<0 or x1>hx:
            directCon = False
            nodeCon = False
        else:
            #First check if we can reach the goal directly.
            if self.collision(x1,y1,self.end[0],self.end[1]):
                directCon = False
            else:
                dst, _ = self.dist_and_angle(x1,y1,self.end[0],self.end[1])

                # Threshold distance from last point to goal (optional)
                if dst > self.stepSize*2:
                    directCon = True # You would set this to false to not have a direct path to the goal.
                else:
                    directCon= True

            if self.collision(x1,y1,x2,y2):
                nodeCon = False
            else:
                nodeCon = True

        return(directCon,nodeCon)

    # return dist and angle b/w two points
    def dist_and_angle(self, x1,y1,x2,y2):
        dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
        angle = math.atan2(y2-y1, x2-x1)
        return(dist,angle)
    
    # b/w two nodes (im lazy)
    def distance(self, n1, n2):
        dist = math.sqrt( ((n2.x-n1.x)**2)+((n2.y-n1.y)**2) )
        return dist
    
    # returns all the nodes within a certain radius
    def near_nodes(self, new_node):
        return [node for node in self.node_list if self.distance(node, new_node) < self.radius]


    def best_parent(self, nx, ny):

        """
            Returns the lowest cost parent (thats how RRT* works)
        """

        best_cost = float('inf')
        best_node = None
        next_node = Nodes(nx, ny)
        for i in range(len(self.node_list)):
            n = self.node_list[i]
            dst = self.distance(n, next_node)
            if dst > self.radius:
                # print("out of radius")
                continue
            cost = n.cost + dst
            if cost<best_cost and not self.collision(n.x, n.y, next_node.x, next_node.y):
                best_cost = cost
                best_node = i
            else:
                pass
                # print("cost", cost, best_cost)
        return best_node, best_cost
    
    # Rewire path to lower cost altertnative
    def rewire(self, new_node, near_nodes):
        for node in near_nodes:
            cost_via_new_node = new_node.cost + self.distance(new_node, node)
            if cost_via_new_node < node.cost and not self.collision(new_node.x , new_node.y, node.x, node.y):
                if node.parent:
                    node.parent.children.remove(node)
                node.parent = new_node
                new_node.children.append(node)
                node.cost = cost_via_new_node
                self.update_children_costs(node)  # Recursively update costs of all children

    def update_children_costs(self, node):
        for child in node.children:
            proposed_cost = node.cost + self.distance(node, child)
            if proposed_cost < child.cost:
                # print("UPDATED CHILD NODE COST")
                child.cost = proposed_cost
                self.update_children_costs(child)  # Recursively update children's costs

    # return the neaerst node index
    def nearest_node(self, x,y):
        temp_dist=[]
        for i in range(len(self.node_list)):
            dist,_ = self.dist_and_angle(x,y,self.node_list[i].x,self.node_list[i].y)
            temp_dist.append(dist)
        return temp_dist.index(min(temp_dist))

    # generate a random point in the occup grid space
    def rnd_point(self, h,l):
        new_y = random.randint(0, h-1)
        new_x = random.randint(0, l-1)
        return (new_x,new_y)


    def RRT(self):

        h,l= self.occugrid.shape # dim of the occu grid

        if self.occugrid[self.start[0], self.start[1]] == self.occup:
            #ie. we are in an obstacle
            return [None, None]
        
        # if self.occugrid[self.end[0], self.end[1]] == 0:
        #     print("END IS COLLISION")
        #     return [None, None]

        if not self.collision(self.start[0], self.start[1], self.end[0], self.end[1]):
            #Just go straight to the end
            return [self.start, self.end]
               
        self.node_list[0] = Nodes(self.start[0],self.start[1])
        self.node_list[0].parent = None
        self.node_list[0].cost = 0

        i=1
        loop = 0
        self.busy = True
        pathFound = False
        while pathFound==False and (not rospy.is_shutdown()):
            # print(f"iter {i}")
            if i > self.max_iters:
                self.busy = False
                return [None, None]

            if loop > 40:
                # print("MAX LOOP")
                return [None, None]

            """
                Pick a random point. Make a branch from the closest node, in the direction of the random point, of a fixed distance.

                Then take the new node and check the fastest way to get there without hitting obstacles. Also check if this node can help reduce costs for other nodes.
            """


            nx,ny = self.rnd_point(h,l)
            
            nearest_ind = self.nearest_node(nx,ny)
            nearest_x = self.node_list[nearest_ind].x
            nearest_y = self.node_list[nearest_ind].y


        
            _,theta = self.dist_and_angle(nearest_x,nearest_y,nx,ny)
            tx=int(nearest_x + self.stepSize*np.cos(theta))
            ty=int(nearest_y + self.stepSize*np.sin(theta))
            
            if ty<0 or ty>self.occugrid.shape[1]-1 or tx<0 or tx>self.occugrid.shape[0]-1:
                loop += 1
                # print("bounds")
                continue

            nearest_ind, new_cost = self.best_parent(tx, ty)

            if nearest_ind is None:
                # print("no best parent")
                loop += 1
                continue
                
            #check direct connection
            directCon,nodeCon = self.check_collision(tx,ty,nearest_x,nearest_y)
            # print("Check collision:",tx,ty,directCon,nodeCon)

            
            if directCon and nodeCon:
                #We can directly go to the end
                self.node_list.append(i)
                self.node_list[i] = Nodes(tx,ty)

                self.node_list[i].parent = self.node_list[nearest_ind]
                self.node_list[nearest_ind].children.append(self.node_list[i])
                self.node_list[i].cost = new_cost

                # self.rewire(self.node_list[i], self.near_nodes(self.node_list[i]))

                # print(f"Path has been found in {i} iterations")

                node = self.node_list[i]
                pos = (node.x, node.y)
                checkpoints = [pos]
                # print("pos", pos)
                while True:
                    pos, node = self.get_parent_coord(node)
                    # print("pos", pos)
                    checkpoints.append(pos)
                    if node is None or pos == self.start:
                        break
                # checkpoints.reverse()
                self.busy = False
                return checkpoints

            elif nodeCon:

                self.node_list.append(i)
                self.node_list[i] = Nodes(tx,ty)
                self.node_list[i].parent = self.node_list[nearest_ind]
                self.node_list[nearest_ind].children.append(self.node_list[i])
                self.node_list[i].cost = new_cost


                #Check to see if you can lower the travel costs.
                self.rewire(self.node_list[i], self.near_nodes(self.node_list[i]))
                i += 1
                loop = 0
                continue

            else:
                loop += 1
                continue
    
    def get_parent_coord(self,node):
        if node.parent is None:
            return self.start, None
        x = node.parent.x 
        y = node.parent.y
        parent = node.parent

        return (x,y), parent