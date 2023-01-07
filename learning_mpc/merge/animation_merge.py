"""
An animation file for the visulization of the environment
"""
import numpy as np
#
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math
from math import cos, sin, tan
import copy
#
from common.vehicle_index import *

#
class SimVisual(object):
    """
    An animation class
    """
    def __init__(self, env):

        self.env = env
        self.lane_len = self.env.lane_len
        self.world_size = self.env.world_size

        self.t_min, self.t_max = 0, env.sim_T
        self.sim_dt = env.sim_dt
        self.vehicle_length, self.vehicle_width = env.vehicle_length, env.vehicle_width
        #self.pivot_point = env.ball_init_pos          # e.g., np.array([2.0, 0.0, 2.0])
        self.frames = []
        #
        # create figure
        self.fig = plt.figure(figsize=(15,15)) # 20 15
        # and figure grid
        #self.gs = gridspec.GridSpec(nrows=4, ncols=10)
        #self.gs = gridspec.GridSpec(nrows=1, ncols=1)
        
        # Create layout of our plots
       # self.ax_2d = plt.figure()
        #
        self.ax_2d = self.fig.add_subplot() # [0:, 3:]
        self.ax_2d.set_xlim([-50, 50]) #self.ax_2d.set_xlim([-1, 1])
        self.ax_2d.set_ylim([-50, 50]) #self.ax_2d.set_ylim([-1, 1])
        self.ax_2d.set_xlabel("x")
        self.ax_2d.set_ylabel("y")
        
        # Plot 2D coordinates,
        self.l_vehicle_pos, = self.ax_2d.plot([], [], 'b-')
        self.l_vehicle_pred_traj, = self.ax_2d.plot([], [], 'r*', markersize=4)

        self.l_vehicle_outline, = self.ax_2d.plot([], [], 'b', linewidth=3)

        self.l_f_v_outline, = self.ax_2d.plot([], [], 'g', linewidth=3)
        #self.l_surrounding_v_outline, = self.ax_2d.plot([], [], 'r', linewidth=2)
        self.l_chance_lf_outline, = self.ax_2d.plot([], [], 'r', linewidth=3)
        self.l_chance_rt_outline, = self.ax_2d.plot([], [], 'r', linewidth=3)
        #self.surrounding_v_pos = self.env.surrounding_v_pos
        #self.surrounding_v_vel = self.env.surrounding_v_vel
        self.chance_pos = self.env.chance_pos
        self.chance_len = self.env.chance_len
        self.chance_wid = self.env.chance_wid

        self.p_high_variable = self.ax_2d.scatter([], [], marker='o', color='g')

        self.l_mainroad_up, = self.ax_2d.plot([-self.world_size,self.world_size], [self.lane_len,self.lane_len], 'black', linewidth=2)
        self.l_mainroad_mid, = self.ax_2d.plot([-self.world_size,self.world_size], [0,0], 'black', linewidth=1)
        self.l_mainroad_dw, = self.ax_2d.plot([-self.world_size,self.world_size], [-self.lane_len,-self.lane_len], 'black', linewidth=2)
        #self.l_mainroad_dw_lf, = self.ax_2d.plot([-self.world_size,-self.lane_len/2*4], [-self.lane_len/2,-self.lane_len/2], 'black', linewidth=2)
        #self.l_mainroad_dw_rt, = self.ax_2d.plot([self.lane_len/2*4,self.world_size], [-self.lane_len/2,-self.lane_len/2], 'black', linewidth=2)

        #
        # #
        self.reset_buffer()
        
    def reset_buffer(self, ):
        #
        self.ts = []
        self.vehicle_pos, self.vehicle_heading, self.vehicle_vel, self.vehicle_cmd = [], [], [], [] #, [] self.vehicle_att, 

    def init_animate(self,):
        
        # Initialize quadrotor 3d trajectory
        self.l_vehicle_pos.set_data([], [])
        # Initialize MPC planned trajectory
        self.l_vehicle_pred_traj.set_data([], [])


        self.l_vehicle_outline.set_data([], [])
        self.l_f_v_outline.set_data([], [])
        #self.l_surrounding_v_outline.set_data([], [])
        self.l_chance_lf_outline.set_data([], [])
        self.l_chance_rt_outline.set_data([], [])
        self.p_high_variable = self.ax_2d.scatter([], [], marker='o', color='y')

        #self.p_high_variable.set_data([],[])


        return self.l_vehicle_pos, self.l_vehicle_pred_traj, \
            self.l_vehicle_outline, self.l_chance_lf_outline, self.l_chance_rt_outline, self.p_high_variable, self.l_f_v_outline

    def update(self, data_info):
        info, t, update = data_info[0], data_info[1], data_info[2]

        vehicle_state = info["vehicle_state"]
        vehicle_act = info["act"]
        chance_pos = info["chance_pos"]
        f_v_pos = info["f_v_pos"]
        pred_vehicle_traj = info["pred_vehicle_traj"]
        #plan_dt = info["plan_dt"]

        
        if update:
            self.reset_buffer()
        else:
            self.ts.append(t)
            #
            self.vehicle_pos.append(vehicle_state[0:2])
            self.vehicle_vel.append(vehicle_state[3:5])
            self.vehicle_cmd.append(vehicle_act[0:2])
            self.vehicle_heading.append(vehicle_state[kphi])

        if len(self.ts) == 0:
            self.init_animate()
        else:

            vehicle_pos_arr = np.array(self.vehicle_pos)
            #
            vehicle_outline = np.array([[-self.vehicle_length/2, self.vehicle_length/2, self.vehicle_length/2, -self.vehicle_length/2,-self.vehicle_length/2,],
                        [self.vehicle_width / 2, self.vehicle_width / 2, - self.vehicle_width / 2, -self.vehicle_width / 2, self.vehicle_width / 2]])
            yaw = self.vehicle_heading[-1]
            Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
            vehicle_outline = (vehicle_outline.T.dot(Rot1)).T
            vehicle_outline[0, :] += vehicle_pos_arr[-1, 0] #vehicle_state[kpx]
            vehicle_outline[1, :] += vehicle_pos_arr[-1:, 1] #vehicle_state[kpy]

            #f_v_outline = copy.deepcopy(vehicle_outline)
            f_v_outline = np.array([[-self.vehicle_length/2, self.vehicle_length/2, self.vehicle_length/2, -self.vehicle_length/2,-self.vehicle_length/2,],
                        [self.vehicle_width / 2, self.vehicle_width / 2, - self.vehicle_width / 2, -self.vehicle_width / 2, self.vehicle_width / 2]])

            f_v_outline[0, :] += f_v_pos[0]
            f_v_outline[1, :] += f_v_pos[1]

            #self.l_vehicle_outline.set_data([np.array(self.vehicle_outline[0, :]).flatten()],[np.array(self.vehicle_outline[1, :]).flatten()])
            self.l_vehicle_outline.set_data([vehicle_outline[0, :]],[vehicle_outline[1, :]])
            self.l_f_v_outline.set_data([ f_v_outline[0, :]],[f_v_outline[1, :]])

            chance_lf_outline = np.array([[-self.world_size, -self.chance_len/2, -self.chance_len/2, -self.world_size,-self.world_size,],
                        [self.chance_wid/2+self.lane_len/2,self.chance_wid/2+self.lane_len/2, - self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, self.chance_wid/2+self.lane_len/2]])

            chance_rt_outline = np.array([[self.chance_len/2, self.world_size, self.world_size, self.chance_len/2,self.chance_len/2,],
                        [self.chance_wid/2+self.lane_len/2,self.chance_wid/2+self.lane_len/2, - self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, self.chance_wid/2+self.lane_len/2]])
            
            chance_lf_outline[0, :] += chance_pos[0]
            #chance_lf_outline[1, :] += chance_pos[1] 

            chance_rt_outline[0, :] += chance_pos[0]
            #chance_rt_outline[1, :] += chance_pos[1] 

            self.l_chance_lf_outline.set_data([chance_lf_outline[0, :]],[chance_lf_outline[1, :]])
            self.l_chance_rt_outline.set_data([chance_rt_outline[0, :]],[chance_rt_outline[1, :]])

            #self.p_high_variable.set_data([self.env.high_variable_pos[0]],[self.env.high_variable_pos[1]])
            self.p_high_variable = self.ax_2d.scatter(self.env.high_variable_pos[0], self.env.high_variable_pos[1], marker='o', color='g')
           
            # plot quadrotor trajectory
            #self.l_vehicle_pos.set_data(vehicle_pos_arr[:, 0], vehicle_pos_arr[:, 1])
            # plot mpc plan trajectory
            self.l_vehicle_pred_traj.set_data(pred_vehicle_traj[:, 0], pred_vehicle_traj[:, 1])


        return  self.l_vehicle_pred_traj, \
                self.l_vehicle_outline, self.l_chance_lf_outline, self.l_chance_rt_outline, self.p_high_variable, self.l_f_v_outline #, self.l_surrounding_v_outline #self.l_vehicle_pos, self.l_vehicle_pred_traj, \
            
    