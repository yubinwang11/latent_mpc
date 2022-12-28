import numpy as np
#
from simulation.vehicle import Bicycle_Dynamics
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

    def __init__(self, mpc, plan_T, plan_dt, init_param = None):
        self.mpc = mpc
        self.plan_T = plan_T
        self.plan_dt = plan_dt

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

        ## Static Chance
        self.chance_pos = [0,0] # [0, 2.0]
        self.chance_len = self.lane_len *2
        self.chance_wid = self.vehicle_width
        self.chance_vel = 0  # 0.5

        ## High_Level Variable
        self.high_variable_pos = None
        
        ## Planner 
        self.sigma = 5 # 10

        ## Reward for RL
        self.reward = 0
        self.tra = True
        self.goal_reward = 300 # sparse positive reward if reach the goal without collision
        self.scale_f = 0.1

        # parameters
        if init_param is None:
            #self.ball_init_pos = np.array([0.0, 0.0, -0.5]) # starting point of the ball
            #self.ball_init_vel = np.array([0.0, -3.5]) # starting velocity of the ball
            self.vehicle_init_pos = np.array([0.0, 0.0]) # starting point of the quadrotor
            self.vehicle_init_vx = np.array([0.0])
        else:
            #self.ball_init_pos = init_param[0] # starting point of the ball
            #self.ball_init_vel = init_param[1] # starting velocity of the ball
            self.vehicle_init_pos = init_param[0]
            self.vehicle_init_heading = init_param[1]
            self.vehicle_init_vx = init_param[2]  # starting point of the quadrotor
       

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
        #self.reset(init_vel=self.ball_init_vel)
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self, goal=None, init_vel=None):
        self.t = 0
        # state for ODE
        self.vehicle_state = self.vehicle.reset(self.vehicle_init_pos, self.vehicle_init_heading, self.vehicle_init_vx)
        #if init_vel is not None:
            #self.ball_state = self.ball.reset(init_vel)
        #else:
            #self.ball_state = self.ball.reset(self.ball_init_vel)
        
        # observation, can be part of the state, e.g., postion
        # or a cartesian representation of the state
        self.goal = goal.tolist()
        self.goal_pos = self.goal[kpx:kpy+1]

        self.obs = []
        self.obs += self.vehicle_state[kpx:kphi+1] # px, py, heading of init pos
        self.obs += self.goal[kpx:kpy+1] # px, py of goal
        self.obs += self.chance_pos[kpx:kpy+1] # px, py of chance pos
        self.obs +=[self.chance_vel]

        #vehicle_obs = self.vehicle_state
        #
        #obs = (quad_obs - ball_obs).tolist()
        #self.obs = vehicle_obs.tolist()
        
        return self.obs

    def step(self, high_variable):
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
            self.reward -= self.scale_f * np.linalg.norm(np.array(self.goal_pos) - np.array(self.vehicle_pos))

        self.chance_pos[0] += self.chance_vel*self.sim_dt
        # simulate one step ball
        #self.ball_state = self.ball.run()
        
        self.obs = []
        self.obs += self.vehicle_state[kpx:kphi+1] # px, py, heading of init pos
        self.obs += self.goal[kpx:kpy+1] # px, py of goal
        self.obs += self.chance_pos[kpx:kpy+1] # px, py of chance pos
        self.obs += [self.chance_vel]

        info = {
            "vehicle_state": self.vehicle_state,
            "act": _act, 
            "pred_vehicle_traj": pred_traj, 
            "plan_dt": self.plan_dt,
            #"cost": cost
            }

        done = False
        if self.t >= (self.sim_T-self.sim_dt) or np.linalg.norm(np.array(self.goal_pos) - np.array(self.vehicle_pos)) < 0.2:
            done = True
            self.reward += self.goal_reward

        return self.obs, self.reward, done, info
    

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

        
    