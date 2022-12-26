"""
Standard MPC for Vehicle Control
"""
from cmath import pi
import casadi as ca
import numpy as np
import time
from os import system
from simulation.vehicle import Bicycle_Dynamics
from common.vehicle_index import *

#
class MPC(object):
    """
    Nonlinear MPC
    """
    def __init__(self, T, dt):
        """
        Nonlinear MPC for vehicle control        
        """

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T/self._dt)

        self.a_max = 1.5; self.a_min = -3 
        self.delta_max = 0.6 ; self.delta_min = -0.6 

        self.vehicle = Bicycle_Dynamics(self._dt)

        # Vehicle constant (kinematics)        (# Quadrotor constant #self._w_max_yaw = 6.0 #self._w_max_xy = 6.0 #self._thrust_min = 2.0 #self._thrust_max = 20.0)
        
        #self.v_max = Vehicle.MAX_SPEED; self.v_min = -self.v_max
        #self.omega_max = BicycleVehicle.MAX_ANGULAR_SPEED/40; self.omega_min = -self.omega_max   
        
        #
        # state dimension (px, py,           # vehicle position
        #                  v,                    # linear velocity
        #                  theta,           # heading angle
        self._s_dim = 6
        # action dimensions (accelaration, steer angle)
        self._u_dim = 2
        
        # cost matrix for tracking the goal point
        self._Q_goal = np.diag([
            10, 10,  # delta_x, delta_y 100 100
            10, # delta_phi 10
            10, 10, # delta_vx delta_vy
            1]) # delta_omega
        
        # cost matrix for the action
        self._Q_u = np.diag([0.1, 0.1]) # a, delta
        self._Q_delta_u = np.diag([1, 1]) # a, delta

        # initial state and control action
        self._vehicle_s0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._vehicle_u0 = [0.0, 0.0]

        self._initDynamics()

    def _initDynamics(self,):
        # # # # # # # # # # # # # # # # # # # 
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # # 

        px, py = ca.SX.sym('px'), ca.SX.sym('py')
        phi = ca.SX.sym('phi')
        vx, vy = ca.SX.sym('vx'), ca.SX.sym('vy')
        #
        omega = ca.SX.sym('omega')
    
        # -- conctenated vector
        self._x = ca.vertcat(px, py, phi, vx, vy, omega) 

        # # # # # # # # # # # # # # # # # # # 
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        a, delta = ca.SX.sym('a'), ca.SX.sym('delta')
        
        # -- conctenated vector
        self._u = ca.vertcat(a, delta)
        
        # # # # # # # # # # # # # # # # # # # 
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        x_dot = ca.vertcat(vx*np.cos(phi) - vy*np.sin(phi),  # f_px
                            vy*np.cos(phi) + vx*np.sin(phi),  # f_py
                            omega, # f_phi
                            a, # f_vx
                            (self.vehicle.Lk*omega - self.vehicle.kf*delta*vx - self.vehicle.m*(vx**2)*omega + vy*(self.vehicle.kf+self.vehicle.kr)) / \
                                (self.vehicle.m*vx - self._dt*(self.vehicle.kf+self.vehicle.kr)), # f_vy
                            (self.vehicle.Lk*vy - self.vehicle.lf*self.vehicle.kf*delta*vx \
                                + omega*((self.vehicle.lf**2)*self.vehicle.kf + (self.vehicle.lr**2)*self.vehicle.kr)) / \
                                    (self.vehicle.Iz*vx - self._dt*((self.vehicle.lf**2)*self.vehicle.kf + (self.vehicle.lr**2)*self.vehicle.kr)) # f_omega
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
        #x_bound = 6 #x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_min[kpy] = -4 + self.vehicle.width/2
        x_max = [x_bound  for _ in range(self._s_dim)]
        x_max[kpy] = 4 - self.vehicle.width/2
        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        #P = ca.SX.sym("P", self._s_dim+(self._s_dim+3)*self._N+self._s_dim)
        self.P = ca.SX.sym("P", self._s_dim+self._s_dim)
        self.X = ca.SX.sym("X", self._s_dim, self._N+1)
        self.U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(self.X[:, :self._N], self.U)
        
        # "Lift" initial conditions
        self.nlp_w += [self.X[:, 0]]
        self.nlp_w0 += self._vehicle_s0
        self.lbw += x_min
        self.ubw += x_max
        
        # # starting point.
        self.nlp_g += [ self.X[:, 0] - self.P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max
        
        for k in range(self._N):
            #
            self.nlp_w += [self.U[:, k]]
            self.nlp_w0 += self._vehicle_u0
            self.lbw += u_min
            self.ubw += u_max
            
            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = P[ idx_k : idx_k_end]

            # cost for tracking the goal position
            cost_goal_k = 0
            # The goal postion.

            delta_s_k = (self.X[:, k+1] - self.P[self._s_dim:])
            cost_goal_k = f_cost_goal(delta_s_k)
            
            if k == 0:
                delta_delta_u_k = self.U[:, k]-[0, 0] #delta_u_k = U[:, k]-[self._gz, 0, 0, 0]
            else:
                delta_delta_u_k = self.U[:, k]- self.U[:, k-1] #delta_u_k = U[:, k]-[self._gz, 0, 0, 0]

            cost_delta_u_k = f_cost_delta_u(delta_delta_u_k)

            delta_u_k = self.U[:, k]-[0, 0] #delta_u_k = U[:, k]-[self._gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_delta_u_k#+  cost_gap_k 

            # New NLP variable for state at end of interval
            self.nlp_w += [self.X[:, k+1]]
            self.nlp_w0 += self._vehicle_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - self.X[:, k+1]]
            self.lbg += g_min
            self.ubg += g_max

            # nlp objective
        self.nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': self.P,               
            'g': ca.vertcat(*self.nlp_g) }        
        
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

    def _initConstraints(self,surr_vehicle_pos):
        nlp_g = self.nlp_g
        lbg = self.lbg 
        ubg =   self.ubg 

        for k in range(self._N):
            dist = (self.X[0, k+1]-surr_vehicle_pos[0])**2 + (self.X[1, k+1]-surr_vehicle_pos[1])**2
            nlp_g += [dist]

            lbg += [self.vehicle.length**2]
            ubg += [np.inf]

        nlp_dict = {'f': self.mpc_obj, 
            'x': ca.vertcat(*self.nlp_w), 
            'p': self.P,               
            'g': ca.vertcat(*nlp_g) }

        return nlp_dict, lbg, ubg      

    def solve(self, ref_states, surr_vehicle_pos):
        
        nlp_dict, lbg, ubg   = self._initConstraints(surr_vehicle_pos)
        # # reload compiled mpc
        #print(self.so_path)
        
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, self.ipopt_options)

        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        self.sol = self.solver(
            x0=self.nlp_w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            p=ref_states, 
            lbg=lbg, 
            ubg=ubg)
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
    
if __name__ == "__main__":
    plan_T = 2.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.1 # Sampling time step for MPC and local planner
    so_path = "./mpc/saved/mpc_v1.so" # saved mpc model (casadi code generation)
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    ref_traj = mpc._vehicle_s0 + np.array([1.5, 1.5, 0.0, 0.0, 0.0, 0.0]).tolist() 
    _act, pred_traj = mpc.solve(ref_traj)
    print(_act)