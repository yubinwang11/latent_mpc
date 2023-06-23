"""
Standard MPC for Vehicle Control
"""
import casadi as ca
import numpy as np
import time
from os import system

#
class High_MPC(object):
    """
    Nonlinear MPC
    """
    def __init__(self, T, dt, L, vehicle_width, lane_width = 4, init_state=None, init_u=None, num_obstacles=9):
        """
        Nonlinear MPC for vehicle control        
        """

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)

        # inter-axle distance
        self.L = L

        self.vehicle_length = self.L #vehicle_width*2
        self.vehicle_width = vehicle_width
        self.lane_width = lane_width
        self.num_obstacles=num_obstacles

        #self.a_max = 1.5; self.a_min = -3
        self.a_max = 1.5*3; self.a_min = -3*3
        self.delta_max = 0.75 ; self.delta_min = -0.75  # 0.75

        self.v_min = 0
        self.v_max = 10

        self.safe_dist = np.sqrt(self.vehicle_length**2+self.vehicle_width**2)

        #
        # state dimension (x, y,           # vehicle position
        #                  v,                    # linear velocity
        #                  phi,           # heading angle
        self._s_dim = 4
        # action dimensions (accelaration, steer angle)
        self._u_dim = 2
        
        # cost matrix for tracking the goal point
        self._Q_goal = np.diag([
            100, 100,  # delta_x, delta_y 100 100 
            100, # delta_phi 10
            10])  # delta_v

        self._Q_u = np.diag([0.1, 0.1]) # a, delta self._Q_u = np.diag([0.1, 0.1]) # a, delta
        self._Q_delta_u = np.diag([1, 1]) # delta_a, delta_steer

        # initial state and control action
        if init_state is None:
            self._vehicle_s0 = [0.0, 0.0, 0.0, 0.0]
        else:
            self._vehicle_s0 = init_state
        if init_u is None:
            self._vehicle_u0 = [0.0, 0.0]
        else:
            self._vehicle_u0 = init_u

        #print(init_state)
        self._initDynamics()

    def _initDynamics(self,):
        # # # # # # # # # # # # # # # # # # # 
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # # 

        x, y = ca.SX.sym('x'), ca.SX.sym('y')
        phi = ca.SX.sym('phi')
        v = ca.SX.sym('v')
        #
        # -- conctenated vector
        self._x = ca.vertcat(x, y, phi, v) 

        # # # # # # # # # # # # # # # # # # # 
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        a, delta = ca.SX.sym('a'), ca.SX.sym('delta')
        
        # -- conctenated vector
        self._u = ca.vertcat(a, delta)
        
        # # # # # # # # # # # # # # # # # # # 
        # --------- System Dynamics/ Kinematics ---------
        # # # # # # # # # # # # # # # # # # #
        # here kinematic bicycle model

        x_dot = ca.vertcat(v*np.cos(phi+delta),  # f_x
                            v*np.sin(phi+delta),  # f_y ------------
                            2*v/self.L*np.sin(delta), # f_phi
                            a, # f_vx
                            )
        '''
        x_dot = ca.vertcat(v*np.cos(phi-delta),  # f_x
                            v*np.sin(phi-delta),  # f_y ------------
                            2*v/self.L*np.sin(-delta), # f_phi
                            a, # f_vx
                            )
        '''

        #
        self.f = ca.Function('f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode'])
                
        # # Fold
        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp") # parallel
        
        # # # # # # # # # # # # # # # 
        # ---- loss function --------
        # # # # # # # # # # # # # # # 

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)

        Delta_u = ca.SX.sym("Delta_u", self._u_dim)
        Delta_delta_u = ca.SX.sym("Delta_delta_u", self._u_dim)      
        
        #        
        cost_goal = Delta_s.T @ self._Q_goal @ Delta_s 
        cost_u = Delta_u.T @ self._Q_u @ Delta_u
        cost_delta_u = Delta_delta_u.T @ self._Q_delta_u @ Delta_delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])
        f_cost_delta_u = ca.Function('cost_delta_u', [Delta_delta_u], [cost_delta_u])

        #
        # # # # # # # # # # # # # # # # # # # # 
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []       # nlp variables
        self.nlp_w0 = []      # initial guess of nlp variables
        self.lbw = []         # lower bound of the variables, lbw <= nlp_x
        self.ubw = []         # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0      # objective 
        self.nlp_g = []       # constraint functions
        self.lbg = []         # lower bound of constrait functions, lbg < g
        self.ubg = []         # upper bound of constrait functions, g < ubg

        u_min = [self.a_min, self.delta_min] #
        u_max = [self.a_max,  self.delta_max] #
        x_bound = np.inf #x_bound = ca.inf

        x_min = [-x_bound for _ in range(self._s_dim)]
        #x_min[0] = 0
        #x_min[1] = -1.5*self.lane_width + self.vehicle_width/2 #-1.5*self.lane_width + self.vehicle_width/2
        x_min[3] = self.v_min

        x_max = [x_bound  for _ in range(self._s_dim)]
        #x_max[0] = 300
        #x_max[1] = 1.5*self.lane_width - self.vehicle_width/2 # - self.vehicle_width/2 #1.5*self.lane_width - self.vehicle_width/2 1.5
        x_max[3] = self.v_max

        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        #P = ca.SX.sym("P", self._s_dim+(self._s_dim+3)*1+self._s_dim)
        #P = ca.SX.sym("P", self._s_dim+self._s_dim)
        self.P = ca.SX.sym("P", self._s_dim+(self.num_obstacles*4)*1+self._s_dim)
        self.X = ca.SX.sym("X", self._s_dim, self._N+1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(self.X[:, :self._N], U)
        
        # "Lift" initial conditions
        self.nlp_w += [self.X[:, 0]]
        self.nlp_w0 += self._vehicle_s0
        self.lbw += x_min
        self.ubw += x_max
        
        # # starting point.
        self.nlp_g += [self.X[:, 0] - self.P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max
        
        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._vehicle_u0
            self.lbw += u_min
            self.ubw += u_max
            
            # cost for tracking the goal position
            cost_goal_k = 0
            #delta_s_k = (X[:, k+1] - P[self._s_dim:]) # The goal postion.
            #cost_goal_k = f_cost_goal(delta_s_k)


            delta_s_k = (self.X[:, k+1] - self.P[self._s_dim+(self.num_obstacles*4)*1:])
            cost_goal_k = f_cost_goal(delta_s_k)

            
            delta_u_k = U[:, k]-[0, 0] #delta_u_k = U[:, k]-[self._gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            if k > 0:
                delta_delta_u_k = U[:, k]-U[:, k-1]
            else:
                delta_delta_u_k = U[:, k]-[0, 0]
            
            cost_delta_u_k = f_cost_delta_u(delta_delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_delta_u_k 

            # New NLP variable for state at end of interval
            self.nlp_w += [self.X[:, k+1]]
            self.nlp_w0 += self._vehicle_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - self.X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

            for i in range(self.num_obstacles):
                self.nlp_g += [ca.sqrt((self.X[0, k+1]-self.P[self._s_dim+i*4]-0.1*self.P[self._s_dim+i*4+3]*(k+1))**2 + \
                                        (self.X[1, k+1]-self.P[self._s_dim+i*4+1])**2)] # k+1 self.vehicle_length

                #self.lbg += [self.safe_dist]
                self.lbg += [3*(0.98**(k))]
                self.ubg += [np.inf]
            
            #self.nlp_g += [self.X[1, k+1]] # k+1
            #self.lbg += [-1.5*self.lane_width+self.vehicle_width]
            #self.ubg += [1.5*self.lane_width-self.vehicle_width]

        
        #print('nlp_g:', self.nlp_g)
        # nlp objective
        
        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': self.P,               
            'g': ca.vertcat(*self.nlp_g)}        
        

        
        # # # # # # # # # # # # # # # # # # # 
        # -- qpoases            
        # # # # # # # # # # # # # # # # # # # 
        # nlp_options ={
        #     'verbose': False, \
        #     "qpsol": "qpoases", \
        #     "hessian_approximation": "gauss-newton", \
        #     "max_iter": 100, 
        #     "tol_du": 1e-2,
        #     "tol_pr": 1e-2,
        #     "qpsol_options": {"sparse":True, "hessian_type": "posdef", "numRefinementSteps":1} 
        # }
        # self.solver = ca.nlpsol("solver", "sqpmethod", nlp_dict, nlp_options)
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared ' + cname + ' -o ' + self.so_path)
        # self.solver = ca.nlpsol("solver", "sqpmethod", self.so_path, nlp_options)
        

        # # # # # # # # # # # # # # # # # # # 
        # -- ipopt
        # # # # # # # # # # # # # # # # # # # 

        self.ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0, 
            "print_time": False
        }
        
        # self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
        # # jit (just-in-time compilation)
        # print("Generating shared library........")
        # cname = self.solver.generate_dependencies("mpc_v1.c")  
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3
        
        # # reload compiled mpc
        #print(self.so_path)
        
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, self.ipopt_options)
        
    def gen_constr(self, obstacles_pos):
        nlp_g = self.nlp_g
        lbg = self.lbg
        ubg = self.ubg

        for k in range(self._N):
            for i in range(self.num_obstacles):
                nlp_g += [ca.sqrt((self.X[0, k]-obstacles_pos[i*2])**2 + (self.X[1, k]-obstacles_pos[i*2+1])**2)] # k+1

                lbg += [self.vehicle_length] #[self.vehicle_length]#**2
                ubg += [np.inf]
        
        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': self.P,               
            'g': ca.vertcat(*nlp_g)}    

        return nlp_dict, lbg, ubg

    def solve(self, ref_states):

        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        
        #nlp_dict, lbg, ubg = self.gen_constr(obstacles_pos)
        #self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, self.ipopt_options) 
        
        self.sol = self.solver(
            x0=self.nlp_w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            p=ref_states, 
            lbg=self.lbg, 
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim+self._u_dim]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self._s_dim+self._u_dim:2*(self._s_dim+self._u_dim)]) + list(sol_x0[self._s_dim+self._u_dim:])
        #print(self.nlp_w0)
        #
        x0_array = np.reshape(sol_x0[:-self._s_dim], newshape=(-1, self._s_dim+self._u_dim))
        
        # return optimal action, and a sequence of predicted optimal trajectory.  
        return opt_u, x0_array
    
    
    def sys_dynamics(self, dt):
        M = 4       # refinement
        DT = dt/M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 =DT*self.f(X, U)
            k2 =DT*self.f(X+0.5*k1, U)
            k3 =DT*self.f(X+0.5*k2, U)
            k4 =DT*self.f(X+k3, U)
            #
            X = X + (k1 + 2*k2 + 2*k3 + k4)/6        
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F
    
