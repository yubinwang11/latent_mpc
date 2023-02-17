import numpy as np
#
from simulation.vehicle import Bicycle_Dynamics
from simulation.object import Surr_Vehicle
from learning_mpc.merge.mpc_merge import High_MPC
#
from common.vehicle_index import *

#
class Space(object):

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)

class MergeEnv(object):

    def __init__(self, curriculum_mode = 'general', eval = False):

        self.curriculum_mode = curriculum_mode
        self.plan_T = 5.0 # Prediction horizon for MPC and local planner
        self.plan_dt = 0.1 # Sampling time step for MPC and local planner

        # simulation parameters ....
        self.sim_T = 15 #7.5 #1.5        # Episode length, seconds
        self.sim_dt = 0.1 #0.02      # simulation time step
        self.max_episode_steps = int(self.sim_T/self.sim_dt)

        # Simulators, a quadrotor and a ball
        self.vehicle = Bicycle_Dynamics(dt=self.sim_dt)
        self.vehicle_length = self.vehicle.length
        self.vehicle_width = self.vehicle.width

        # Road Parameters
        self.world_size = 120 # 80
        self.lane_len = 6

        self.curriculum_mode = curriculum_mode

        ## High_Level Variable
        self.high_variable_pos = None
        
        ## Planner 
        self.sigma = 1 # 10
    
        self.action_space = Space(
            low=np.array([-3.0, -0.6]), #low=np.array([-3.0]),
            high=np.array([1.5, 0.6]) #high=np.array([2*self.plan_T])
        )

        # reset the environment
        self.eval = eval
        self.end_con = np.inf

        self.mpc = None
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self):
        self.t = 0
        # state for ODE
        self.vehicle_state = self.vehicle.reset(curriculum_mode=self.curriculum_mode)

        self.initial_u = [0, 0]

        if self.curriculum_mode == 'general':
            # Sampling range of the chance's initial position
            self.c_xy_reldist = np.array(
                [ [10, 30]]   # x
            )
            # Sampling range of the chance's initial velocity
            self.c_vxy_dist = np.array(
                [ [2.0, 5.0]  # vx
                ] 
            )
            # Sampling range of the front vehicle's initial position
            self.f_v_relxy_dist = np.array(
                [ [25, 35]]   # x
            )
            # Sampling range of the front vehicle's initial velocity
            self.f_v_vxy_dist = np.array(
                [ [0.5, 3]  # vx
                ] 
            )

            # Chance Parameters
            self.chance_pos = [self.vehicle_state[kpx]+np.random.uniform(
                    low=self.c_xy_reldist[0, 0], high=self.c_xy_reldist[0, 1]),self.lane_len/2] # [0, 2.0]
            
            self.end_con = np.pi
            
        elif self.curriculum_mode == 'easy':
            # Sampling range of the chance's initial position
            self.c_xy_reldist = np.array(
                [ [15, 55]]   # x
            )
            # Sampling range of the chance's initial velocity
            self.c_vxy_dist = np.array(
                [ [0, 0]  # vx
                ] 
            )
            # Sampling range of the front vehicle's initial position
            self.f_v_relxy_dist = np.array(
                [ [15, 25]]   # x
            )
            # Sampling range of the front vehicle's initial velocity
            self.f_v_vxy_dist = np.array(
                [ [0.5, 3]  # vx
                ] 
            )

            # Chance Parameters
            self.chance_pos = [self.vehicle_state[kpx]+np.random.uniform(
                    low=self.c_xy_reldist[0, 0], high=self.c_xy_reldist[0, 1]),self.lane_len/2] # [0, 2.0]
            self.end_con = np.pi / 4

        elif self.curriculum_mode == 'medium':
            # Sampling range of the chance's initial position
            self.c_xy_reldist = np.array(
                [ [15, 40]]   # x
            )
            # Sampling range of the chance's initial velocity
            self.c_vxy_dist = np.array(
                [ [0.5, 2]  # vx
                ] 
            )
            # Sampling range of the front vehicle's initial position
            self.f_v_relxy_dist = np.array(
                [ [15, 25]]   # x
            )
            # Sampling range of the front vehicle's initial velocity
            self.f_v_vxy_dist = np.array(
                [ [0.5, 3]  # vx
                ] 
            )

            # Chance Parameters
            self.chance_pos = [self.vehicle_state[kpx]+np.random.uniform(
                    low=self.c_xy_reldist[0, 0], high=self.c_xy_reldist[0, 1]),self.lane_len/2] # [0, 2.0]
            self.end_con = np.pi / 3
        
        elif self.curriculum_mode == 'hard':
            # Sampling range of the chance's initial position
            self.c_xy_reldist = np.array(
               [ [10, 30]
                  ]   # x
            )
            # Sampling range of the chance's initial velocity
            self.c_vxy_dist = np.array(
                [ [2, 4]  # vx
                ] 
            )
            # Sampling range of the front vehicle's initial position
            self.f_v_relxy_dist = np.array(
                [ [25, 35]]   # x
            )
            # Sampling range of the front vehicle's initial velocity
            self.f_v_vxy_dist = np.array(
                [ [0.5, 3]  # vx
                ] 
            )

            # Chance Parameters
            self.chance_pos = [self.vehicle_state[kpx]+np.random.uniform(
                    low=self.c_xy_reldist[0, 0], high=self.c_xy_reldist[0, 1]),self.lane_len/2] # [0, 2.0]
            self.end_con = np.pi/1.2
            
        self.chance_vel = np.random.uniform(
                low=self.c_vxy_dist[0, 0], high=self.c_vxy_dist[0, 1])
        
        self.chance_len = self.lane_len *2 #+ self.vehicle_length
        self.chance_wid = self.vehicle_width

        self.f_v_pos = [max(self.chance_pos[0],self.vehicle_state[kpx])+np.random.uniform(
                low=self.f_v_relxy_dist[0, 0], high=self.f_v_relxy_dist[0, 1]), -self.lane_len/2]
        #self.f_v_pos = [np.random.uniform(low=self.f_v_xy_dist[0, 0], high=self.f_v_xy_dist[0, 1]), -self.lane_len/2]
        self.f_v_vel = np.random.uniform(
                low=self.f_v_vxy_dist[0, 0], high=self.f_v_vxy_dist[0, 1])

        #self.goal = np.array([2, 3.0, 0, 3.0, 0.0, 0.0])
        self.goal = np.array([30,self.lane_len/2, 0, self.chance_vel, 0.0, 0.0])

        self.goal = self.goal.tolist()
        self.goal_pos = self.goal[kpx:kpy+1]

        self.obs = []
        self.obs += self.vehicle_state[kpx:kphi+1] # px, py, heading of init pos
        self.obs += self.goal[kpx:kpy+1] # px, py of goal
        self.obs += self.chance_pos[kpx:kpy+1] # px, py of chance pos
        self.obs +=[self.chance_vel]

        self.collided = False
        self.success = False
        
        return self.obs

    def step(self, high_variable, step_i):
        
        value_Qmax = high_variable[6:12]

        if self.mpc is None: 
            if (self.eval):
                self.mpc = High_MPC(T=self.plan_T, dt=self.plan_dt, Qmax = value_Qmax, lane_len = self.lane_len, init_state=self.vehicle_state, init_u=self.initial_u,stimulate=False)
            else:
                self.mpc = High_MPC(T=self.plan_T, dt=self.plan_dt, Qmax = value_Qmax, lane_len = self.lane_len, init_state=self.vehicle_state, init_u=self.initial_u)
        else:
            pass

        reward = 0
        self.t += self.sim_dt
        
        #self.high_variable_pos = high_variable[kpx:kpy+1]

        print("===========================================================")    
        #
        vehicle_state = self.vehicle_state
        goal_state = self.goal
        
        current_t = self.t

        tra_state = high_variable[kpx:6] + [current_t, high_variable[-1], self.sigma]

        ref_traj = vehicle_state + tra_state + goal_state

        # ------------------------------------------------------------
        # run  model predictive control
        _act, pred_traj = self.mpc.solve(ref_traj)

        self.vehicle_state = self.vehicle.run(_act)
        self.vehicle_pos = self.vehicle_state[kpx:kpy+1]

        self.chance_pos[0] += self.chance_vel*self.sim_dt
        self.goal[0] = self.chance_pos[0]+self.vehicle_length/2

        self.f_v_pos[0] += self.f_v_vel*self.sim_dt
        
        self.surr_v_left_pos = np.array([(self.chance_pos[0]-self.chance_len/2-0)/2, self.chance_pos[1]])
        self.surr_v_right_pos = np.array([(self.world_size+self.chance_pos[0]+self.chance_len/2)/2, self.chance_pos[1]])
        #self.surr_v_left_pos = np.array([(self.chance_pos[0]+self.world_size)/2, self.chance_pos[1]])
        self.surr_v_left = Surr_Vehicle(position=self.surr_v_left_pos, heading=np.array(0), vel=self.chance_vel, length=np.array(self.chance_pos[0]-self.chance_len/2+0), width=self.chance_wid)
        self.surr_v_right = Surr_Vehicle(position=self.surr_v_right_pos, heading=np.array(0), vel=self.chance_vel, length=np.array(self.world_size-self.chance_pos[0]-self.chance_len/2), width=self.chance_wid)
        self.f_v = Surr_Vehicle(position=np.array([self.f_v_pos]), heading=np.array(0), vel=self.f_v_vel, length=self.vehicle_length, width=self.vehicle_width)

        collision = self.vehicle._is_colliding(self.surr_v_left) or self.vehicle._is_colliding(self.surr_v_right) or self.vehicle._is_colliding(self.f_v) 
        # sparse reward for collision avoidance 
        if (collision):
            self.collided = collision

            if self.vehicle_state[kvx] >= 0:
                reward -= 5
            else:
                reward -= 2
        
        if step_i == 0:
            out_of_road_up, out_of_road_down = self._check_out_of_road(high_variable[kpy])
            if (out_of_road_up):
                reward -= 2* np.linalg.norm(high_variable[kpy]- (self.lane_len - self.vehicle_length/2))
            elif  (out_of_road_down):
                reward -= 2* np.linalg.norm(high_variable[kpy]- (-self.lane_len + self.vehicle_length/2))

            if high_variable[-1] > self.sim_T or high_variable[-1] < 0:
                reward -= 5 * min(abs(high_variable[-1]-self.sim_T), abs(high_variable[-1]-0))
            
            #if high_variable[-1] <= 0:
                #reward -= 5* abs(high_variable[-1])
            #else:
            reward -= abs(high_variable[-1])

            if high_variable[2] > np.pi/2 or high_variable[2] < -np.pi/2:
                reward -= 5 * min(abs(high_variable[2]-np.pi/2),abs(high_variable[2]-(-np.pi/2)))

            if np.array(high_variable[0]) <= np.array(self.chance_pos[0] + self.vehicle_length/2):
                dist_x = np.linalg.norm(np.array(high_variable[0])-np.array(self.chance_pos[0]+ self.vehicle_length/2))
                reward -= dist_x

            #if high_variable[3] > 3 * self.chance_vel or high_variable[3] < 0:
                #reward -= 3 * min(abs(high_variable[3]-0),abs(high_variable[3]- 3* self.chance_vel))

        # observation
        self.obs = []
        self.obs += self.vehicle_state[kpx:kphi+1] # px, py, heading of init pos
        self.obs += self.goal[kpx:kpy+1] # px, py of goal
        self.obs += self.chance_pos[kpx:kpy+1] # px, py of chance pos
        self.obs += [self.chance_vel]
        self.obs += self.f_v_pos[kpx:kpy+1]
        self.obs += [self.f_v_vel]

        info = {
            "vehicle_state": self.vehicle_state,
            "chance_pos": self.chance_pos,
            "f_v_pos": self.f_v_pos,
            "act": _act, 
            "pred_vehicle_traj": pred_traj, 
            "current_t": current_t,
            "plan_dt": self.plan_dt,
            "surr_v_left": self.surr_v_left,
            "surr_v_right": self.surr_v_right,
            "high_variable": high_variable

            #"cost": cost
            }

        done = False
        if np.linalg.norm(np.array(self.goal[0:3]) - np.array(self.vehicle_state)[0:3]) < self.end_con: #1.25
            done = True
            reward += 100

            if self.collided == False:
                self.success = True

            #dist_x = np.linalg.norm(np.array(high_variable[0])-np.array(self.chance_pos[0]))
            #reward -= dist_x
            
        elif self.t >= (self.sim_T-self.sim_dt):
            done = True
        
        if (done):
            reward -= abs(current_t)

        return self.obs, reward, done, info
    
    def _check_out_of_road(self, py_z):

        out_of_road_up = False
        out_of_road_down = False
        if py_z >= self.lane_len - self.vehicle_length/2:
            out_of_road_up = True
        elif py_z <= -self.lane_len + self.vehicle_length/2:
            out_of_road_down = True

        return out_of_road_up, out_of_road_down

        
    