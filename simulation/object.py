import numpy as np
#from scipy.spatial.transform import Rotation as R
from common.vehicle_index import *
#
"""
"""
class Surr_Vehicle(object):
    #
    def __init__(self, position=np.array([0,0]), heading=np.array(0), vel= np.array([0]), length=np.array([4]), width=np.array([2])):

        self.position = position
        self.heading = heading
        self.length = length
        self.width = width
        self.vel = vel

        self.diagonal = np.sqrt(self.length**2 + self.width**2)

    def polygon(self):
        points = np.array([
            [-self.length / 2, -self.width / 2],
            [-self.length / 2, +self.width / 2],
            [+self.length / 2, +self.width / 2],
            [+self.length / 2, -self.width / 2],
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])