import numpy as np
import casadi as ca
#from scipy.spatial.transform import Rotation as R
from common.vehicle_index import *
from common.utils import project_polygon
from common.utils import are_polygons_intersecting
#
"""
Standard Vehicle Dynamics with Bicycle Model
"""
class Bicycle_Dynamics(object):
    #
    def __init__(self, dt):
        self.s_dim = 6
        self.a_dim = 2
        #
        self._state = np.zeros(shape=self.s_dim) 
        self._actions = np.zeros(shape=self.a_dim)

        #
        self._gz = 9.81
        self._dt = dt# self._arm_l = 0.3   # m
        
        ## Vehicle Parameter settings
        self.length = 4 # 4.5
        self.width =2 #
        self.diagonal = np.sqrt(self.length**2 + self.width**2)

        self.kf = -128916
        self.kr = -85944
        self.lf = 1.06
        self.lr = 1.85
        self.m = 1412
        self.Iz = 1536.7
        self.Lk = (self.lf*self.kf) - (self.lr*self.kr)

        #
        #self.reset()
        # self._t = 0.0
    
    def reset(self, position=None, heading=None,  vx = None, curriculum_mode='general'):
        
        if curriculum_mode == 'general':
            # Sampling range of the vehicle's initial position
            self._xy_dist = np.array(
                [ [5, 35 ]]   # x
            )
            # Sampling range of the vehicle's initial velocity
            self._vxy_dist = np.array(
                [ [0.0, 10.0]  # vx
                ] 
            )

        self._state = np.zeros(shape=self.s_dim) 
        if position is None:
            # self._state[kQuatW] = 1.0 #         
            #
            # initialize position,  not randomly
            self._state[kpx] = np.random.uniform(
                low=self._xy_dist[0, 0], high=self._xy_dist[0, 1])
           # self._state[kpx]  = 0    
            self._state[kpy] = -3 #np.random.uniform(
                #low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1])
            #self._state[kPosZ] = np.random.uniform(
                #low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1])
            
            # initialize heading
            #quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
            # normalize the quaternion
            #self._state[kQuatW:kQuatZ+1] = quad_quat0 / np.linalg.norm(quad_quat0)
            self._state[kphi] = 0
            
            # initialize velocity, not randomly
            self._state[kvx] = np.random.uniform(
                low=self._vxy_dist[0, 0], high=self._vxy_dist[0, 1])
           # self._state[kvx] = 0
            #self._state[kVelY] = np.random.uniform(
                #low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1])
            self._state[kvy] = 0
            self._state[komega] = 0
            #
        else:
            self._state[kpx] = position[0]
            self._state[kpy] = position[1]
            # heading
            self._state[kphi] = heading
            # v
            self._state[kvx] = vx
            self._state[kvy] = 0
            
            # initialize angular velocity
            self._state[komega] = 0
            #

        return self._state.tolist()
    
    def run(self, action):
        """
        Apply the control command on the vehicle and transits the system to the next state
        """
        # rk4 int
        M = 4
        DT = self._dt / M
        #
        X = self._state
        for i in range(M):
            k1 = DT*self._f(X, action)
            k2 = DT*self._f(X + 0.5*k1, action)
            k3 = DT*self._f(X + 0.5*k2, action)
            k4 = DT*self._f(X + k3, action)
            #
            X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0
        #
        self._state = X
        #print(f"real state is {self._state}")

        self.position = np.array([self._state[kpx], self._state[kpy]])
        self.heading = np.array(self._state[kphi])

        return self._state.tolist()

    def _f(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        px, py, phi, vx, vy, omega = state                    
        a, delta = action
        #
        dstate = np.zeros(shape=self.s_dim)

        dstate[kpx] = vx*np.cos(phi) - vy*np.sin(phi)
        dstate[kpy] = vy*np.cos(phi) + vx*np.sin(phi)
        dstate[kphi] = omega; dstate[kvx] = a
        dstate[kvy] = (self.Lk*omega - self.kf*delta*vx - self.m*(vx**2)*omega + vy*(self.kf+self.kr)) / (self.m*vx - self._dt*(self.kf+self.kr))
        dstate[komega] = (self.Lk*vy - self.lf*self.kf*delta*vx + omega*((self.lf**2)*self.kf + (self.lr**2)*self.kr)) / \
                                    (self.Iz*vx - self._dt*((self.lf**2)*self.kf + (self.lr**2)*self.kr))


        return dstate

    def _is_colliding(self, other):
        # Fast spherical pre-check
        #if np.linalg.norm(other.position - self.position) > (self.diagonal + other.diagonal) / 2: 
            #return False,
        # Accurate rectangular check
        #else:
        return are_polygons_intersecting(self.polygon(), other.polygon())
    
    def polygon(self):
        points = np.array([
            [-self.length / 2, -self.width / 2],
            [-self.length / 2, +self.width / 2],
            [+self.length / 2, +self.width / 2],
            [+self.length / 2, -self.width / 2],
        ]).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([
            [c, -s],
            [s, c]
        ])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])