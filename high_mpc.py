"""
Standard MPC for Vehicle Control
"""
import casadi as ca
import numpy as np
import time
from os import system
from common.vehicle_index import *

#
class High_MPC(object):
    """
    Nonlinear MPC
    """
    def __init__(self, T, dt, L, vehicle_width, lane_width = 4, init_state=None, init_u=None):
        """
        Nonlinear MPC for vehicle control        
        """

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)

        # inter-axle distance
        self.L = L

        self.vehicle_width = vehicle_width
        self.lane_width = lane_width

        #self.a_max = 1.5 * 1.5 ; self.a_min = -3 * 1.5
        self.a_max = 1.5*3; self.a_min = -3*3
        self.delta_max = 0.75 ; self.delta_min = -0.75 

        self.v_min = -5
        self.v_max = 10

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
        
        self._Q_tra = np.diag([
            500, 500,  # delta_x, delta_y 100 100
            50, # delta_phi
            50]) #  delta_v

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
                            v*np.sin(phi+delta),  # f_y
                            2*v/self.L*np.sin(delta), # f_phi
                            a, # f_vx
                            )
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
        Delta_p = ca.SX.sym("Delta_p", self._s_dim)

        Delta_u = ca.SX.sym("Delta_u", self._u_dim)
        Delta_delta_u = ca.SX.sym("Delta_delta_u", self._u_dim)      
        
        #        
        cost_goal = Delta_s.T @ self._Q_goal @ Delta_s 
        cost_tra = Delta_p.T @ self._Q_tra @ Delta_p 
        cost_u = Delta_u.T @ self._Q_u @ Delta_u
        cost_delta_u = Delta_delta_u.T @ self._Q_delta_u @ Delta_delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_tra = ca.Function('cost_tra', [Delta_p], [cost_tra])
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
        x_min[0] = 0
        x_min[1] = -1.5*self.lane_width + self.vehicle_width/2
        x_min[3] = self.v_min

        x_max = [x_bound  for _ in range(self._s_dim)]
        x_max[0] = 300
        x_max[1] = 1.5*self.lane_width - self.vehicle_width/2
        x_max[3] = self.v_max

        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        P = ca.SX.sym("P", self._s_dim+(self._s_dim+3)*1+self._s_dim)
        #P = ca.SX.sym("P", self._s_dim+self._s_dim)
        X = ca.SX.sym("X", self._s_dim, self._N+1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(X[:, :self._N], U)
        
        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._vehicle_s0
        self.lbw += x_min
        self.ubw += x_max
        
        # # starting point.
        self.nlp_g += [ X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max
        
        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._vehicle_u0
            self.lbw += u_min
            self.ubw += u_max
            
            # retrieve time constant
            #idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            idx_k = self._s_dim + self._s_dim
            #idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)\
            idx_k_end = self._s_dim+self._s_dim+3

            time_k = P[ idx_k : idx_k_end]
            

            # # # # # # # # # # # # # # # # # # # # # # # # 
            # - compute exponetial weights
            # - time_k[2] defines the temporal spread of the weight
            # - time_k[0] defines the current time 
            # - time_k[1] defines the best traversal time, which is selected via 
            #              a high-level policy / a deep high-level policy

            # # # # # # # # # # # # # # # # # # # # # # # # 
            #weight = ca.exp(- time_k[2] * (time_k[0]-time_k[1])**2 ) 
            weight = ca.exp(- time_k[2] * (time_k[0]-time_k[1])**2 ) 
            
            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0
            #delta_s_k = (X[:, k+1] - P[self._s_dim:]) # The goal postion.
            #cost_goal_k = f_cost_goal(delta_s_k)

            
            #if k >= self._N-1: # The goal postion.
            if k >= self._N -1: # The goal postion.

                delta_s_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*1:])
                cost_goal_k = f_cost_goal(delta_s_k)

            else:

                delta_s_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*1:])
                cost_goal_k = f_cost_goal(delta_s_k)
                                    
                # cost for tracking the moving gap
                delta_p_k = (X[0:self._s_dim, k+1] - P[self._s_dim+(self._s_dim+3)*0 : \
                    self._s_dim+(self._s_dim+3)*(0+1)-3]) 
                cost_tra_k = f_cost_tra(delta_p_k) * weight
                #print(cost_gap_k.shape)
            
            delta_u_k = U[:, k]-[0, 0] #delta_u_k = U[:, k]-[self._gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            if k > 0:
                delta_delta_u_k = U[:, k]-U[:, k-1]
            else:
                delta_delta_u_k = U[:, k]-[0, 0]
            
            cost_delta_u_k = f_cost_delta_u(delta_delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_delta_u_k  +  cost_tra_k 

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k+1]]
            self.nlp_w0 += self._vehicle_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': P,               
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

        ipopt_options = {
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
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

    def solve(self, ref_states):

        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
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
    
