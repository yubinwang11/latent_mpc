import numpy as np
#
from simulation.vehicle import Bicycle_Dynamics
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

    def __init__(self):

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
        self.world_size = 30
        self.lane_len = 6
        self.surrounding_v_pos = [-20,self.lane_len/2]
        self.surrounding_v_vel = 2
        self.surrounding_v_state = self.surrounding_v_pos.append(self.surrounding_v_vel)

        # Sampling range of the vehicle's initial position
        self.c_xy_dist = np.array(
            [ [-15, 0]]   # x
        )
        # Sampling range of the vehicle's initial velocity
        self.c_vxy_dist = np.array(
            [ [3.0, 5.0]  # vx
            ] 
        )

        # Chance Parameters
        self.chance_pos = [-10,self.lane_len/2] # [0, 2.0]
        self.chance_pos[0] = np.random.uniform(
                low=self.c_xy_dist[0, 0], high=self.c_xy_dist[0, 1])

        self.chance_vel = np.random.uniform(
                low=self.c_vxy_dist[0, 0], high=self.c_vxy_dist[0, 1])
        self.chance_len = self.lane_len *2
        self.chance_wid = self.vehicle_width
        #self.chance_vel = 2  # 0.5

        ## High_Level Variable
        self.high_variable_pos = None
        
        ## Planner 
        self.sigma = 5 # 10

        ## Reward for RL
        #self.reward = 0
        self.tra = True
        self.goal_reward = 300 # sparse positive reward if reach the goal without collision
        self.scale_f = 0.1

        #self.goal = np.array([2, 3.0, 0, 3.0, 0.0, 0.0])
        self.goal = np.array([-10,self.lane_len/2, 0, self.chance_vel, 0.0, 0.0])

        self.vehicle_init_pos = np.array([-25.0, -3.0])
        self.vehicle_init_heading = np.array([0])
        self.vehicle_init_vx = np.array([3.0])

        initial_state = [self.vehicle_init_pos[0], self.vehicle_init_pos[1], self.vehicle_init_heading, self.vehicle_init_vx, 0 , 0]
        initial_u = [0.2, 0]
        self.mpc = High_MPC(T=self.plan_T, dt=self.plan_dt, init_state=initial_state, init_u=initial_u)
       
        # state space
        self.observation_space = Space(
            low=np.array([-30.0, -30.0, -2*np.pi, -30.0, -30.0, -30.0, -30.0, -10.0]), #low=np.array([-10.0, -10.0, -10.0, -2*np.pi, -2*np.pi, -2*np.pi, -10.0, -10.0, -10.0]),
            high=np.array([30.0, 30.0,    2*np.pi, 30.0, 30.0, 30.0, 30.0, 10.0]), #high=np.array([10.0, 10.0, 10.0, 2*np.pi, 2*np.pi, 2*np.pi, 10.0, 10.0, 10.0]),
        )
        #self.obs = None

        self.action_space = Space(
            low=np.array([-3.0, -0.6]), #low=np.array([-3.0]),
            high=np.array([1.5, 0.6]) #high=np.array([2*self.plan_T])
        )

        # reset the environment
        self.t = 0
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self):
        self.t = 0
        # state for ODE
        #self.vehicle_state = self.vehicle.reset(self.vehicle_init_pos, self.vehicle_init_heading, self.vehicle_init_vx)
        self.vehicle_state = self.vehicle.reset()

        # observation, can be part of the state, e.g., postion
        # or a cartesian representation of the state
        self.goal = self.goal.tolist()
        self.goal_pos = self.goal[kpx:kpy+1]

        self.obs = []
        self.obs += self.vehicle_state[kpx:kphi+1] # px, py, heading of init pos
        self.obs += self.goal[kpx:kpy+1] # px, py of goal
        self.obs += self.chance_pos[kpx:kpy+1] # px, py of chance pos
        self.obs +=[self.chance_vel]
        
        return self.obs

    def step(self, high_variable):
        
        reward = 0
        self.t += self.sim_dt
        
        self.high_variable_pos = high_variable[kpx:kpy+1]

        print("===========================================================")    
        #
        vehicle_state = self.vehicle_state
        goal_state = self.goal
        
        current_t = self.t
        tra_state = high_variable[kpx:kphi+1] + [current_t, high_variable[-1], self.sigma]
        #print("current tra_state: ", tra_state)
        ref_traj = vehicle_state + tra_state + goal_state

        # ------------------------------------------------------------
        # run  model predictive control
        _act, pred_traj = self.mpc.solve(ref_traj)
        #print(len(pred_traj[0]))
    
        self.vehicle_state = self.vehicle.run(_act)
        self.vehicle_pos = self.vehicle_state[kpx:kpy+1]

        if (self.tra):
            reward -= self.scale_f * np.linalg.norm(np.array(self.goal_pos) - np.array(self.vehicle_pos))

        self.chance_pos[0] += self.chance_vel*self.sim_dt
        self.goal[0] = self.chance_pos[0]+self.vehicle_length/2
        # simulate one step ball
        #self.ball_state = self.ball.run()
        
        self.obs = []
        self.obs += self.vehicle_state[kpx:kphi+1] # px, py, heading of init pos
        self.obs += self.goal[kpx:kpy+1] # px, py of goal
        self.obs += self.chance_pos[kpx:kpy+1] # px, py of chance pos
        self.obs += [self.chance_vel]

        info = {
            "vehicle_state": self.vehicle_state,
            "chance_pos": self.chance_pos,
            "act": _act, 
            "pred_vehicle_traj": pred_traj, 
            "plan_dt": self.plan_dt,
            #"cost": cost
            }

        done = False
        if self.t >= (self.sim_T-self.sim_dt) or np.linalg.norm(np.array(self.goal_pos) - np.array(self.vehicle_pos)) < 0.2:
            done = True
            reward += self.goal_reward

        return self.obs, reward, done, info
    

    def _is_within_gap(gap_corners, point):
        A, B, C = [], [], []    
        for i in range(len(gap_corners)):
            p1 = gap_corners[i]
            p2 = gap_corners[(i + 1) % len(gap_corners)]
            
            # calculate A, B and C
            a = -(p2.y - p1.y)
            b = p2.x - p1.x
            c = -(a * p1.x + b * p1.y)

            A.append(a)
            B.append(b)
            C.append(c)
        D = []
        for i in range(len(A)):
            d = A[i] * point.x + B[i] * point.y + C[i]
            D.append(d)

        t1 = all(d >= 0 for d in D)
        t2 = all(d <= 0 for d in D)
        return t1 or t2

        
    