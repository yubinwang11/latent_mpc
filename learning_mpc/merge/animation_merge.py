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
        #self.fig = plt.figure(figsize=(15,15)) # 20 15、
        self.fig = plt.figure(figsize=(12,12))
        # and figure grid
        self.gs = gridspec.GridSpec(nrows=3, ncols=2)
        #self.gs = gridspec.GridSpec(nrows=1, ncols=1)
        
        # Create layout of our plots
        #
        self.ax_2d = self.fig.add_subplot(self.gs[:2, :])
        #self.ax_2d = self.fig.add_subplot()
        self.ax_2d.set_xlim([12, 87]) #self.ax_2d.set_xlim([-1, 1])
        self.ax_2d.set_ylim([-25, 25]) #self.ax_2d.set_ylim([-1, 1])
        self.ax_2d.set_xlabel("x")
        self.ax_2d.set_ylabel("y")
        
        self.act_min = -6; self.act_max = 3
        self.t_min, self.t_max = 0, env.sim_T

        self.ax_act = self.fig.add_subplot(self.gs[2, :])
        self.ax_act.set_ylim([self.act_min, self.act_max])
        self.ax_act.set_xlim([0, self.t_max/2-1.5])

        self.l_acc, = self.ax_act.plot([], [], '-b', label='acceleration', linewidth=3)
        self.l_steer, = self.ax_act.plot([], [], '-r', label='steer angle', linewidth=3)
        self.ax_act.legend( handles=[self.l_acc, self.l_steer], fontsize=25, loc=1)

        '''
        self.pos_min, self.pos_max = 0, 80

        self.ax_pos = self.fig.add_subplot(self.gs[2, 1])
        self.ax_act.set_ylim([self.pos_min, self.pos_max])
        self.ax_act.set_xlim([0, self.t_max])

        self.l_pos, = self.ax_act.plot([], [], '-k', label='acceleration')
        self.l_steer, = self.ax_act.plot([], [], '-r', label='steer angle')
        self.ax_act.legend( handles=[self.l_acc, self.l_steer], fontsize=13, loc=1)
        '''


        # Plot 2D coordinates,
        self.l_vehicle_pos, = self.ax_2d.plot([], [], 'k-', linewidth=3)
        self.l_vehicle_pred_traj, = self.ax_2d.plot([], [], 'r*', markersize=4)

        self.l_vehicle_outline, = self.ax_2d.plot([], [], 'b', linewidth=4)

        self.l_f_v_outline, = self.ax_2d.plot([], [], 'g', linewidth=4)
        #self.l_surrounding_v_outline, = self.ax_2d.plot([], [], 'r', linewidth=2)
        self.l_trafficflow_left, = self.ax_2d.plot([], [], 'g', linewidth=4)
        self.l_trafficflow_right, = self.ax_2d.plot([], [], 'g', linewidth=4)
        #self.surrounding_v_pos = self.env.surrounding_v_pos
        #self.surrounding_v_vel = self.env.surrounding_v_vel
        self.chance_pos = self.env.chance_pos
        self.chance_len = self.env.chance_len
        self.chance_wid = self.env.chance_wid

        self.p_high_variable = self.ax_2d.scatter([], [],  marker='*', color='brown')

        self.l_mainroad_up, = self.ax_2d.plot([0,self.world_size], [self.lane_len,self.lane_len], 'black', linewidth=3)
        self.l_mainroad_mid, = self.ax_2d.plot([0,self.world_size], [0,0], 'black', linewidth=2, linestyle='dashed')
        self.l_mainroad_dw, = self.ax_2d.plot([0,self.world_size], [-self.lane_len,-self.lane_len], 'black', linewidth=3)
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

        self.l_trafficflow_left.set_data([], [])
        self.l_trafficflow_right.set_data([], [])

        self.l_acc.set_data([], [])
        self.l_steer.set_data([], [])

        #self.p_high_variable.set_data([],[])

        return self.l_vehicle_pos, self.l_vehicle_pred_traj, \
            self.l_vehicle_outline, self.p_high_variable, self.l_f_v_outline, \
            self.l_trafficflow_left, self.l_trafficflow_right

    def update(self, data_info):
        info, t, update = data_info[0], data_info[1], data_info[2]

        vehicle_state = info["vehicle_state"]
        vehicle_act = info["act"]
        chance_pos = info["chance_pos"]
        f_v_pos = info["f_v_pos"]
        pred_vehicle_traj = info["pred_vehicle_traj"]
        surr_v_left = info["surr_v_left"]
        surr_v_right = info["surr_v_right"]
        high_variable = info["high_variable"]

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
            
            vehicle_act_arr = np.array(self.vehicle_cmd)
            self.l_acc.set_data(self.ts, vehicle_act_arr[:, 0])
            self.l_steer.set_data(self.ts, vehicle_act_arr[:, 1])

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

            trafficflow_gap_len = self.vehicle_length + 3
            left_trafficflow_len = abs(self.chance_pos[0] - self.chance_len/2)
            left_trafficflow_vehicle_num = int(left_trafficflow_len // trafficflow_gap_len)
            right_trafficflow_len = abs(self.world_size - (self.chance_pos[0] + self.chance_len/2))
            right_trafficflow_vehicle_num = int(right_trafficflow_len // trafficflow_gap_len)

            trafficflow_left_outline = []
            for num_left in range(left_trafficflow_vehicle_num):
                center_left = (self.chance_pos[0] - self.chance_len/2) - self.vehicle_length/2
                center_left -= num_left * trafficflow_gap_len
                trafficflow_left_vehicle = np.array([[center_left+self.vehicle_length/2, center_left+self.vehicle_length/2, center_left-self.vehicle_length/2, center_left-self.vehicle_length/2,center_left+self.vehicle_length/2,center_left+self.vehicle_length/2,center_left-self.vehicle_length/2,center_left-self.vehicle_length/2,],
                        [self.lane_len/2, -self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, self.lane_len/2]])
                trafficflow_left_outline.append(trafficflow_left_vehicle)
                #self.l_trafficflow_left_fill = plt.fill(np.array(trafficflow_left_vehicle[0]), np.array(trafficflow_left_vehicle[1]), color ='g')

            trafficflow_left_outline = np.array(trafficflow_left_outline)
            #self.l_trafficflow_left.set_data([trafficflow_left_outline[:,0, :]],[trafficflow_left_outline[:,1, :]])
            self.l_trafficflow_left.set_data([trafficflow_left_outline[:,0, :]],[trafficflow_left_outline[:,1, :]])

            trafficflow_right_outline = []
            for num_right in range(right_trafficflow_vehicle_num):
                center_right = (self.chance_pos[0] + self.chance_len/2) + self.vehicle_length/2
                center_right += num_right * trafficflow_gap_len
                trafficflow_right_vehicle = np.array([[center_right-self.vehicle_length/2, center_right-self.vehicle_length/2, center_right+self.vehicle_length/2, center_right+self.vehicle_length/2,center_right-self.vehicle_length/2,center_right-self.vehicle_length/2,center_right+self.vehicle_length/2,center_right+self.vehicle_length/2,],
                        [self.lane_len/2, +self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, self.lane_len/2]])
                trafficflow_right_outline.append(trafficflow_right_vehicle)
                #self.l_trafficflow_left_fill = plt.fill(np.array(trafficflow_left_vehicle[0]), np.array(trafficflow_left_vehicle[1]), color ='g')

            trafficflow_right_outline = np.array(trafficflow_right_outline)
            #self.l_trafficflow_left.set_data([trafficflow_left_outline[:,0, :]],[trafficflow_left_outline[:,1, :]])
            self.l_trafficflow_right.set_data([trafficflow_right_outline[:,0, :]],[trafficflow_right_outline[:,1, :]])

    
            #self.p_high_variable.set_data([self.env.high_variable_pos[0]],[self.env.high_variable_pos[1]])
            self.p_high_variable = self.ax_2d.scatter(high_variable[0], high_variable[1], marker='*', color='brown', s=300)
           
            # plot quadrotor trajectory
            self.l_vehicle_pos.set_data(vehicle_pos_arr[:, 0], vehicle_pos_arr[:, 1])

            # plot mpc plan trajectory
            self.l_vehicle_pred_traj.set_data(pred_vehicle_traj[:, 0], pred_vehicle_traj[:, 1])

            # save eps fig
            #plt.savefig('./1.pdf', dpi=300)
            self.fig.savefig('./1.png', dpi=600)

        return  self.l_vehicle_pred_traj, \
                self.l_vehicle_outline, self.p_high_variable, self.l_f_v_outline,\
                self.l_vehicle_pos, self.l_trafficflow_left, self.l_trafficflow_right, \
                self.l_acc, self.l_steer
                #self.l_trafficflow_left_fill
                #, self.l_surrounding_v_outline #self.l_vehicle_pos, self.l_vehicle_pred_traj, \
            
    