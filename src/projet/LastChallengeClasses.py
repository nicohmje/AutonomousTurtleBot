import numpy as np
import math


class Bottle:
    def __init__(self, index, position=None, gate=None, colour=None):
        self.index = index
        self.position = position
        self.gate = gate
        self.colour = colour

    def get_position(self):
        return (self.position[0], self.position[1])
    
    def set_position(self, position:tuple):
        self.position = position

    def set_colour(self, colour:int):
        self.colour = colour

    def get_colour(self):
        return self.colour

    def set_gate(self, gate:int):
        self.gate = gate

    def get_gate(self):
        return self.gate
    
    def get_index(self):
        return self.index
        


class Gate:
    def __init__(self, center_position=None, bottle_index1=None, bottle_index2=None, colour=None):
        self.center_position = center_position
        self.bottle_index1 = bottle_index1
        self.bottle_index2 = bottle_index2

        self.offset_point1 = None
        self.offset_point2 = None

        self.confirmed = False
        self.colour = colour

        # print("New gate alert! :", self.bottle_index1,self.bottle_index2)

    def confirm(self):
        self.confirmed = True

    def get_center_pos(self):
        return self.center_position
    
    def set_colour(self, colour:int, bottles:list):
        bottles[self.bottle_index1].set_colour(colour)
        bottles[self.bottle_index2].set_colour(colour)

        self.colour = colour
    
    def get_colour(self):
        return self.colour
    
    def get_bottles_indices(self):
        return self.bottle_index1, self.bottle_index2
    

    def get_offset_points(self):
        return self.offset_point1, self.offset_point2

    def update(self, bottles:list):
        c1 = bottles[self.bottle_index1].get_position()
        c2 = bottles[self.bottle_index2].get_position()

        vec = np.array(c1) - np.array(c2)
        perp_vec = np.array([-vec[1], vec[0]])
        unit_perp_vec = perp_vec / np.linalg.norm(perp_vec)

        offset = 0.24 #Self.sim 0.24

        cX = np.mean([c1[0],c2[0]])
        cY = np.mean([c1[1],c2[1]])

        self.center_position = (cX, cY)

        self.offset_point1 = np.array(self.center_position) + offset * unit_perp_vec
        self.offset_point2 = np.array(self.center_position) - offset * unit_perp_vec
        