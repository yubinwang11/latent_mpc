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
import scipy
import os
from pathlib import Path
import matplotlib.transforms as mtransforms
import matplotlib as mpl
#
from common.vehicle_index import *
from PIL import Image  
import random

#
class SimVisual(object):
    """
    An animation class
    """
    def __init__(self, env):
        
        self.surr_v_color_list = ['blue.png','orange.png','green.png']
        self.f_v_img = Image.open('orange.png')
        self.ego_v_img = Image.open('red.png')
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
        self.fig = plt.figure(figsize=(20,20))
        # and figure grid
        self.gs = gridspec.GridSpec(nrows=10, ncols=1)
        #self.gs = gridspec.GridSpec(nrows=1, ncols=1)
        
        # Create layout of our plots
        #
        ax_2d = self.fig.add_subplot(self.gs[:5, :])
        ax_2d.grid(True)
        #self.ax_2d = self.fig.add_subplot()
        ax_2d.set_xlim([20, 72]) #self.ax_2d.set_xlim([-1, 1])
        ax_2d.set_ylim([-13, 13]) #self.ax_2d.set_ylim([-1, 1]) 25 25
        ax_2d.set_xlabel("${p_x} (m)$", fontsize=20)
        ax_2d.set_ylabel("${p_y} (m)$", fontsize=20)
        plt.xticks(size = 20) # ontproperties = 'Times New Roman', 
        plt.yticks(size = 20)

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
        #self.l_vehicle_pos, = self.ax_2d.plot([], [], 'royalblue', linewidth=3)
        #self.l_vehicle_pred_traj, = self.ax_2d.plot([], [], 'darkorange', marker = '*',  markersize=5)
        #self.l_vehicle_pred_traj, = self.ax_2d.plot([], [], color = 'darkorange', marker = '*',  markersize=5)
        self.l_vehicle_pred_traj, = ax_2d.plot([], [], 'r*',  markersize=7)

        self.l_vehicle_outline, = ax_2d.plot([], [], 'royalblue', linewidth=3)

        self.l_f_v_outline, = ax_2d.plot([], [], 'darkorange', linewidth=3)
        self.l_trafficflow_left, = ax_2d.plot([], [], 'darkorange', linewidth=3)
        self.l_trafficflow_right, = ax_2d.plot([], [], 'darkorange', linewidth=3)

        self.chance_pos = self.env.chance_pos
        self.chance_len = self.env.chance_len
        self.chance_wid = self.env.chance_wid

        #self.p_high_variable = self.ax_2d.scatter([], [],  marker='*', color='brown')

        self.reset_buffer()
        
    def reset_buffer(self, ):
        #
        self.ts = []
        self.vehicle_pos, self.vehicle_heading, self.vehicle_vel, self.vehicle_cmd = [], [], [], [] #, [] self.vehicle_att, 
        self.vehicle_vx, self.vehicle_vy = [], []

    def init_animate(self,):
        
        # Initialize quadrotor 3d trajectory
        #self.l_vehicle_pos.set_data([], [])
        # Initialize MPC planned trajectory
        self.l_vehicle_pred_traj.set_data([], [])

        self.l_vehicle_outline.set_data([], [])
        #self.l_vehicle_fill.set_data([], [])
        self.l_f_v_outline.set_data([], [])
        #self.l_surrounding_v_outline.set_data([], [])

        self.l_trafficflow_left.set_data([], [])
        self.l_trafficflow_right.set_data([], [])

        #self.l_acc.set_data([], [])
        #self.l_steer.set_data([], [])

        #self.l_vx.set_data([], [])
        #self.l_vy.set_data([], [])

        #self.p_high_variable.set_data([],[])
        #self.l_vehicle_fill = self.ax_2d.fill([],[], facecolor='g', alpha=0.5)

        return self.l_vehicle_pred_traj, \
            self.l_vehicle_outline, self.l_f_v_outline, \
            self.l_trafficflow_left, self.l_trafficflow_right, \
            #self.l_acc, self.l_steer, self.l_vx, self.l_vy
            #self.l_vehicle_fill, 
    
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
        current_t = info["current_t"]
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
            self.vehicle_vx.append(vehicle_state[kvx])
            self.vehicle_vy.append(vehicle_state[kvy])

        if len(self.ts) == 0:
            self.init_animate()
        else:
            self.fig = plt.figure(figsize=(15,15))
            self.gs = gridspec.GridSpec(nrows=10, ncols=1)
            ax_2d = self.fig.add_subplot(self.gs[:5, :])

            ax_2d.grid(True)
            ax_2d.set_xlim([12, 64]) #self.ax_2d.set_xlim([-1, 1]) 20 72
            ax_2d.set_ylim([-13, 13]) #self.ax_2d.set_ylim([-1, 1]) 25 25 -13 13
            ax_2d.set_xlabel("${p_x} (m)$", fontsize=20)
            ax_2d.set_ylabel("${p_y} (m)$", fontsize=20)
            plt.xticks(size = 20) # ontproperties = 'Times New Roman', 
            plt.yticks(size = 20)
        
            self.l_mainroad_up, = ax_2d.plot([0,self.world_size], [self.lane_len,self.lane_len], 'black', linewidth=3)
            self.l_mainroad_mid, = ax_2d.plot([0,self.world_size], [0,0], 'black', linewidth=2, linestyle='dashed')
            self.l_mainroad_dw, = ax_2d.plot([0,self.world_size], [-self.lane_len,-self.lane_len], 'black', linewidth=3)

            self.l_vehicle_pred_traj, = ax_2d.plot([], [], 'r*',  markersize=7)
            self.l_vehicle_pred_traj.set_data([], [])

            self.art = ax_2d.scatter([],[],c=[])
            self.cax = self.fig.add_subplot(self.gs[5, :])
            self.cmap = mpl.cm.winter
            self.norm = mpl.colors.Normalize(vmin=-3, vmax=15)
            self.cb = mpl.colorbar.ColorbarBase(self.cax, cmap=self.cmap,  norm=self.norm, orientation='horizontal') 
            self.cax.set_xlabel("${v_x} (m/s)$", fontsize=20)
            plt.xticks(size = 20) # ontproperties = 'Times New Roman', 
            plt.yticks(size = 20)

            self.ax_speed = self.fig.add_subplot(self.gs[6:8, :])
            self.ax_speed.grid(True)
            self.ax_speed.set_ylim([-5, 15])
            self.ax_speed.set_xlim([0, self.t_max/2+1])
            plt.xticks(size = 20) # ontproperties = 'Times New Roman', 
            plt.yticks(size = 20)

            self.l_vx, = self.ax_speed.plot([], [], 'royalblue', label='${v_x}$', linewidth=3)
            self.l_vy, = self.ax_speed.plot([], [], 'darkorange', label='${v_y}$', linewidth=3)
            self.ax_speed.legend( handles=[self.l_vx, self.l_vy], fontsize=25, loc=0)

            self.act_min = -6; self.act_max = 3
            self.t_min, self.t_max = 0, self.env.sim_T

            self.ax_act = self.fig.add_subplot(self.gs[8:10, :])
            self.ax_act.grid(True)
            self.ax_act.set_ylim([self.act_min-5, self.act_max+5])
            self.ax_act.set_xlim([0, self.t_max/2+2])
            plt.xticks(size = 20) # ontproperties = 'Times New Roman', 
            plt.yticks(size = 20)

            self.ax_speed.set_xlabel("Time ($s$)", fontsize=20)
            self.ax_speed.set_ylabel("Speed ($m/s$)", fontsize=20)


            self.l_acc, = self.ax_act.plot([], [], 'crimson', label='${a} (m/s^2)$ ', linewidth=3)
            self.l_steer, = self.ax_act.plot([], [], 'goldenrod', label='${\delta} (rad)$', linewidth=3)
            self.ax_act.legend( handles=[self.l_acc, self.l_steer], fontsize=25, loc=4)

            self.ax_act.set_xlabel("Time ($s$)", fontsize=20)
            self.ax_act.set_ylabel("Actions", fontsize=20)

            self.l_acc.set_data([], [])
            self.l_steer.set_data([], [])

            self.l_vx.set_data([], [])
            self.l_vy.set_data([], [])

            vehicle_act_arr = np.array(self.vehicle_cmd)
            self.l_acc.set_data(self.ts, vehicle_act_arr[:, 0])
            self.l_steer.set_data(self.ts, vehicle_act_arr[:, 1])

            self.l_vx.set_data(self.ts, self.vehicle_vx)
            self.l_vy.set_data(self.ts, self.vehicle_vy)


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
            #self.l_vehicle_outline.set_data([vehicle_outline[0, :]],[vehicle_outline[1, :]])
            #self.ego_v_img = self.ego_v_img.rotate(yaw*57.3,expand=True)
            #self.ego_v_img = scipy.ndimage.rotate(self.ego_v_img,yaw*57.3)
            #self.ax_2d.clear(self.ego_v)
            self.ego_v = ax_2d.imshow(self.ego_v_img, interpolation='none',
                                                                origin='lower',  extent=(vehicle_pos_arr[-1, 0]-self.vehicle_length/1.8,vehicle_pos_arr[-1, 0]+self.vehicle_length/1.8,\
                                                                vehicle_pos_arr[-1, 1]-self.vehicle_width/1.8,vehicle_pos_arr[-1, 1]+self.vehicle_width/1.8), clip_on=True)
            #self.ego_v = self.ax_2d.imshow(self.ego_v_img.rotate(yaw*57.3,expand=True),  extent=(vehicle_pos_arr[-1, 0]-self.vehicle_length/1.3,vehicle_pos_arr[-1, 0]+self.vehicle_length/1.3,\
                                                                #vehicle_pos_arr[-1, 1]-self.vehicle_width/1.3,vehicle_pos_arr[-1, 1]+self.vehicle_width/1.3))
            #print(yaw)
            trans_data =  mtransforms.Affine2D().rotate_deg_around(vehicle_pos_arr[-1, 0], vehicle_pos_arr[-1, 1], np.rad2deg(yaw)) + ax_2d.transData
            self.ego_v.set_transform(trans_data)

            self.f_v = ax_2d.imshow(self.f_v_img, extent=(f_v_pos[0]-self.vehicle_length/1.8,f_v_pos[0]+self.vehicle_length/1.8,-self.vehicle_width/1.8-self.lane_len/2,+self.vehicle_width/1.8-self.lane_len/2))

            trafficflow_gap_len = self.vehicle_length + 3.5
            left_trafficflow_len = abs(self.chance_pos[0] - self.chance_len/2)
            left_trafficflow_vehicle_num = int(left_trafficflow_len // trafficflow_gap_len)
            right_trafficflow_len = abs(self.world_size - (self.chance_pos[0] + self.chance_len/2))
            right_trafficflow_vehicle_num = int(right_trafficflow_len // trafficflow_gap_len)

            trafficflow_left_outline = []
            self.flow_v_list = []
            for num_left in range(left_trafficflow_vehicle_num):
                center_left = (self.chance_pos[0] - self.chance_len/2) - self.vehicle_length/2
                center_left -= num_left * trafficflow_gap_len
                trafficflow_left_vehicle = np.array([[center_left+self.vehicle_length/2, center_left+self.vehicle_length/2, center_left-self.vehicle_length/2, center_left-self.vehicle_length/2,center_left+self.vehicle_length/2,center_left+self.vehicle_length/2,center_left-self.vehicle_length/2,center_left-self.vehicle_length/2,],
                        [self.lane_len/2, -self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, self.lane_len/2]])
                trafficflow_left_outline.append(trafficflow_left_vehicle)
                #self.l_trafficflow_left_fill = plt.fill(np.array(trafficflow_left_vehicle[0]), np.array(trafficflow_left_vehicle[1]), color ='g')
                #self.flow_v_left=self.ax_2d.imshow(self.flow_v_img, extent=(center_left-self.vehicle_length/1.3,center_left+self.vehicle_length/1.3,-self.vehicle_width/1.3+self.lane_len/2,+self.vehicle_width/1.3+self.lane_len/2))
                img_path = random.choice(self.surr_v_color_list)
                self.flow_v_img = Image.open(img_path)
                self.flow_v = ax_2d.imshow(self.flow_v_img, extent=(center_left-self.vehicle_length/1.8,center_left+self.vehicle_length/1.8,-self.vehicle_width/1.8+self.lane_len/2,+self.vehicle_width/1.8+self.lane_len/2))
                self.flow_v_list.append(self.flow_v)

            #trafficflow_left_outline = np.array(trafficflow_left_outline)
            #self.l_trafficflow_left.set_data([trafficflow_left_outline[:,0, :]],[trafficflow_left_outline[:,1, :]])
            #self.l_trafficflow_left.set_data([trafficflow_left_outline[:,0, :]],[trafficflow_left_outline[:,1, :]])

            trafficflow_right_outline = []
            for num_right in range(right_trafficflow_vehicle_num):
                center_right = (self.chance_pos[0] + self.chance_len/2) + self.vehicle_length/2
                center_right += num_right * trafficflow_gap_len
                trafficflow_right_vehicle = np.array([[center_right-self.vehicle_length/2, center_right-self.vehicle_length/2, center_right+self.vehicle_length/2, center_right+self.vehicle_length/2,center_right-self.vehicle_length/2,center_right-self.vehicle_length/2,center_right+self.vehicle_length/2,center_right+self.vehicle_length/2,],
                        [self.lane_len/2, +self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, -self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, +self.chance_wid/2+self.lane_len/2, self.lane_len/2]])
                trafficflow_right_outline.append(trafficflow_right_vehicle)
                #self.l_trafficflow_left_fill = plt.fill(np.array(trafficflow_left_vehicle[0]), np.array(trafficflow_left_vehicle[1]), color ='g')
                img_path = random.choice(self.surr_v_color_list)
                self.flow_v_img = Image.open(img_path)
                self.flow_v = ax_2d.imshow(self.flow_v_img, extent=(center_right-self.vehicle_length/1.8,center_right+self.vehicle_length/1.8,-self.vehicle_width/1.8+self.lane_len/2,+self.vehicle_width/1.8+self.lane_len/2))
                self.flow_v_list.append(self.flow_v)

            #trafficflow_right_outline = np.array(trafficflow_right_outline)
            #self.l_trafficflow_left.set_data([trafficflow_left_outline[:,0, :]],[trafficflow_left_outline[:,1, :]])
            #self.l_trafficflow_right.set_data([trafficflow_right_outline[:,0, :]],[trafficflow_right_outline[:,1, :]])

            #self.p_high_variable.set_data([self.env.high_variable_pos[0]],[self.env.high_variable_pos[1]])
            #self.p_high_variable = self.ax_2d.scatter(high_variable[0], high_variable[1], marker='*', color='brown', s=300)
           
            # plot quadrotor trajectory
            self.l_vehicle_pos = ax_2d.scatter(vehicle_pos_arr[:, 0], vehicle_pos_arr[:, 1], c=self.vehicle_vx, cmap='winter', edgecolors='none')
            data = np.hstack((vehicle_pos_arr[:, 0][:,np.newaxis], vehicle_pos_arr[:, 1][:,np.newaxis]))
            self.art.set_offsets(data)
            self.art.set_color(self.cmap(self.norm(self.vehicle_vx))) 
            #self.fig.colorbar(mpl.cm.ScalarMappable(), cax=self.ax_2d, orientation='horizontal')
            #self.sub_colorbar = self.fig.colorbar(self.l_vehicle_pos, ax=self.ax_2d)
            #self.l_vehicle_pos.set_data(vehicle_pos_arr[:, 0], vehicle_pos_arr[:, 1])

            # plot mpc plan trajectory
            self.l_vehicle_pred_traj.set_data(pred_vehicle_traj[:, 0], pred_vehicle_traj[:, 1])

            # save eps fig
            #plt.savefig('./1.pdf', dpi=300)
            eval_dir = Path('./figs')
            fig_name = '%i' % (current_t*10)
            plt.tight_layout()
            self.fig.savefig(eval_dir / fig_name, dpi=600)

            ax_2d.clear()

        if len(self.flow_v_list) == 13:
            return  self.l_vehicle_pred_traj,  \
                    self.l_vehicle_pos, self.art, \
                    self.l_acc, self.l_steer, self.l_vx, self.l_vy, \
                    self.ego_v,\
                    self.f_v, self.flow_v_list[0],self.flow_v_list[1],self.flow_v_list[2], self.flow_v_list[3],self.flow_v_list[4], \
                    self.flow_v_list[5], self.flow_v_list[6], self.flow_v_list[-6],self.flow_v_list[-5], self.flow_v_list[-4], self.flow_v_list[-3], self.flow_v_list[-2] , self.flow_v_list[-1]     #self.flow_v_left_list.all
                #self.l_vehicle_fill
                #self.l_trafficflow_left_fill
                #, self.l_surrounding_v_outline #self.l_vehicle_pos, self.l_vehicle_pred_traj, \ self.p_high_variable, 
        
        elif len(self.flow_v_list) == 14:
            return  self.l_vehicle_pred_traj, \
                    self.l_vehicle_pos, self.art, \
                    self.l_acc, self.l_steer, self.l_vx, self.l_vy, \
                    self.ego_v,\
                    self.f_v, self.flow_v_list[0],self.flow_v_list[1],self.flow_v_list[2], self.flow_v_list[3],self.flow_v_list[4], \
                    self.flow_v_list[5], self.flow_v_list[6], self.flow_v_list[7], self.flow_v_list[-6],self.flow_v_list[-5], self.flow_v_list[-4], self.flow_v_list[-3], self.flow_v_list[-2] , self.flow_v_list[-1]
        
        else:
            return  self.l_vehicle_pred_traj, \
                    self.l_vehicle_pos,  self.art, \
                    self.l_acc, self.l_steer, self.l_vx, self.l_vy, \
                    self.ego_v,\
                    self.f_v, self.flow_v_list[0],self.flow_v_list[1],self.flow_v_list[2], self.flow_v_list[3],self.flow_v_list[4], \
                    self.flow_v_list[5], self.flow_v_list[6], self.flow_v_list[7], self.flow_v_list[8],self.flow_v_list[-6],self.flow_v_list[-5], self.flow_v_list[-4], self.flow_v_list[-3], self.flow_v_list[-2] , self.flow_v_list[-1]
        
    