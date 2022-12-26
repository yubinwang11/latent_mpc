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
        self.fig = plt.figure(figsize=(self.world_size, 15)) # 20 15
        # and figure grid
        #self.gs = gridspec.GridSpec(nrows=4, ncols=10)
        self.gs = gridspec.GridSpec(nrows=1, ncols=1)
        
        # Create layout of our plots
        
        #
        self.ax_2d = self.fig.add_subplot(self.gs[0:, 0:]) # [0:, 3:]
        self.ax_2d.set_xlim([-30, 30]) #self.ax_2d.set_xlim([-1, 1])
        self.ax_2d.set_ylim([-30, 30]) #self.ax_2d.set_ylim([-1, 1])
        self.ax_2d.set_xlabel("x")
        self.ax_2d.set_ylabel("y")
        
        # Plot 2D coordinates,
        self.l_vehicle_pos, = self.ax_2d.plot([], [], 'b-')
        self.l_vehicle_pred_traj, = self.ax_2d.plot([], [], 'r*', markersize=4)
        #self.l_ball_pred_traj, = self.ax_3d.plot([], [], [], 'k*', markersize=0)
        #
        #self.l_ball, = self.ax_3d.plot([], [], [], 'ko')
        #self.l_vehicle_front, = self.ax_2d.plot([], [], 'b', linewidth=2)
        #self.l_vehicle_back, = self.ax_2d.plot([], [], 'b', linewidth=2)
        #self.l_vehicle_left, = self.ax_2d.plot([], [], 'g', linewidth=2)
        #self.l_vehicle_right, = self.ax_2d.plot([], [], 'g', linewidth=2)
        self.l_vehicle_outline, = self.ax_2d.plot([], [], 'b', linewidth=5)
        self.l_surr_v_outline, = self.ax_2d.plot([], [], 'r', linewidth=5)
        self.surr_v_pos = self.env.surr_v_pos
        #
        #self.l_vehicle_x, = self.ax_2d.plot([], [], 'r', linewidth=3)
        #self.l_vehicle_y, = self.ax_2d.plot([], [], 'g', linewidth=3)
        
        #
        #self.ax_3d.scatter(self.pivot_point[0], self.pivot_point[1], self.pivot_point[2], marker='o', color='g')
        # Draw a circle on the x=0 'wall'
        self.l_road_upper, = self.ax_2d.plot([-self.world_size,self.world_size], [self.lane_len,self.lane_len], 'black', linewidth=3)
        self.l_road_mid, = self.ax_2d.plot([-self.world_size,self.world_size], [0,0], 'black', linewidth=2, linestyle = 'dashdot')
        self.l_road_down, = self.ax_2d.plot([-self.world_size,self.world_size], [-self.lane_len,-self.lane_len], 'black', linewidth=3)
        # # Ground
        # width, height = 5, 2
        # g = Rectangle(xy=(0.5-width, 0-height), width=2*width, height=2*height, \
        #     alpha=0.8, facecolor='gray', edgecolor='black')
        # self.ax_3d.add_patch(g)
        # art3d.pathpatch_2d_to_3d(g, z=0, zdir="z")
        #
        # #
        self.reset_buffer()
        
    def reset_buffer(self, ):
        #
        self.ts = []
        self.vehicle_pos, self.vehicle_heading, self.vehicle_vel, self.vehicle_cmd = [], [], [], [] #, [] self.vehicle_att, 
        #self.ball_pos, self.ball_vel, self.ball_att = [], [], []
        #self.vehicle_hist = []

    def init_animate(self,):
        
        # Initialize quadrotor 3d trajectory
        self.l_vehicle_pos.set_data([], [])
        # Initialize MPC planned trajectory
        self.l_vehicle_pred_traj.set_data([], [])

        # Initialize quad arm
        #self.l_vehicle_x.set_data([], [])
        #self.l_vehicle_x.set_2d_properties([])
        #self.l_vehicle_y.set_data([], [])
        #self.l_vehicle_y.set_2d_properties([])
        #self.l_vehicle_front.set_data([], [])
        #self.l_vehicle_back.set_data([], [])
        #self.l_vehicle_left.set_data([], [])
        #self.l_vehicle_right.set_data([], [])
        self.l_vehicle_outline.set_data([], [])
        self.l_surr_v_outline.set_data([], [])


        return self.l_vehicle_pos, self.l_vehicle_pred_traj, \
            self.l_vehicle_outline, self.l_surr_v_outline #self.l_vehicle_front, self.l_vehicle_back, self.l_vehicle_left, self.l_vehicle_right
            #self.l_vehicle_x, self.l_vehicle_y,
               

    def update(self, data_info):
        info, t, update = data_info[0], data_info[1], data_info[2]

        vehicle_state = info["vehicle_state"]
        vehicle_act = info["act"]
        pred_vehicle_traj = info["pred_vehicle_traj"]
        plan_dt = info["plan_dt"]
        
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

            #self.l_vehicle_outline.set_data([np.array(self.vehicle_outline[0, :]).flatten()],[np.array(self.vehicle_outline[1, :]).flatten()])
            self.l_vehicle_outline.set_data([vehicle_outline[0, :]],[vehicle_outline[1, :]])

            #self.surrounding_v_pos[0] += self.surrounding_v_vel * self.sim_dt
            surr_v_outline = np.array([[-self.vehicle_length/2, self.vehicle_length/2, self.vehicle_length/2, -self.vehicle_length/2,-self.vehicle_length/2,],
                        [self.vehicle_width / 2, self.vehicle_width / 2, - self.vehicle_width / 2, -self.vehicle_width / 2, self.vehicle_width / 2]])
            surr_v_outline[0,:] += self.surr_v_pos[0]
            surr_v_outline[1,:] += self.surr_v_pos[1]
            self.l_surr_v_outline.set_data([surr_v_outline[0, :]],[surr_v_outline[1, :]])

            
            # plot quadrotor trajectory
            #self.l_vehicle_pos.set_data(vehicle_pos_arr[:, 0], vehicle_pos_arr[:, 1])
            # plot mpc plan trajectory
            self.l_vehicle_pred_traj.set_data(pred_vehicle_traj[:, 0], pred_vehicle_traj[:, 1])
            #if vehicle_pos_arr[-1, 0] <= 2.0:
                # plot planner trajectory
                #self.l_ball_pred_traj.set_data(np.array([pred_ball_traj[opt_idx, 0]]), np.array([pred_ball_traj[opt_idx, 1]]))
                #self.l_ball_pred_traj.set_3d_properties(np.array([pred_ball_traj[opt_idx, 2]]))

        return  self.l_vehicle_pred_traj, \
                self.l_vehicle_outline, self.l_surr_v_outline #self.l_vehicle_pos, self.l_vehicle_pred_traj, \
            
            #self.l_vehicle_front, self.l_vehicle_back, \
            #self.l_vehicle_left, self.l_vehicle_right  #\
            #self.l_vehicle_x, self.l_vehicle_y,
    