import random
import numpy as np
import carla
#
from high_mpc import High_MPC
from common.vehicle_index import *

def get_longitudinal_speed(vehicle):
    velocity = vehicle.get_velocity()
    forward_vector = vehicle.get_transform().get_forward_vector()
    longitudinal_speed = np.dot(np.array([velocity.x, -velocity.y, velocity.z]), np.array([forward_vector.x,  -forward_vector.y, forward_vector.z]))

    return longitudinal_speed

def get_state_frenet(vehicle, map):

    x = map.get_waypoint(vehicle.get_location(), project_to_road=True).s
    centerline_waypoint= map.get_waypoint_xodr(34, -2,x) # road and lane id
    tangent_vector = centerline_waypoint.transform.get_forward_vector()
    normal_vector = carla.Vector3D(-(-tangent_vector.y), tangent_vector.x, 0)
    normal_vector_normalized = np.array([normal_vector.x, -normal_vector.y]) /  np.linalg.norm(np.array([normal_vector.x, -normal_vector.y]))
    y_hat = np.array([vehicle.get_location().x-centerline_waypoint.transform.location.x, 
                                    -vehicle.get_location().y-(-centerline_waypoint.transform.location.y)])
    y = np.dot(normal_vector_normalized, y_hat)
    forward_angle = np.arctan2(-tangent_vector.y, tangent_vector.x) * 180/np.pi
    if -180 <= forward_angle <= 0:
        forward_angle += 360
    global_yaw = -vehicle.get_transform().rotation.yaw
    if -180 <= global_yaw <= 0:
        global_yaw += 360
    yaw = (forward_angle - global_yaw)/180 * np.pi
    speed = get_longitudinal_speed(vehicle)
    vehicle_state =np.array([x, y, yaw, speed]).tolist()

    return  vehicle_state

def get_control_input(acceleration, steer_angle, dead_zone=0.1):

    max_throttle=0.75; max_brake=0.5; max_steering=0.75; KP=0.75 # 0.1

    if acceleration >= dead_zone:
        throttle = min(max_throttle, KP*acceleration)
        brake = 0
    elif acceleration <= -dead_zone:
        throttle = 0 
        brake = min(max_brake, -KP*acceleration)
    else:
        throttle = 0
        brake = 0.1

    desired_steer_angle = np.clip(steer_angle, -1, 1)

    control = carla.VehicleControl(throttle=throttle, brake=brake, steer=desired_steer_angle, hand_brake=False)

    return control

def spawn_autopilot_agent(blueprint_lib, world, spawn_transform):

    agent_bp = random.choice(blueprint_lib.filter('vehicle.*'))
    agent_bp.set_attribute('role_name', 'autopilot')
    agent = world.spawn_actor(agent_bp, spawn_transform)
    agent.set_autopilot(True)

    return agent

class Env(object):

    def __init__(self, world):

        self.world = world
        self.map = self.world.get_map()

        # create the blueprint library
        self.blueprint_library = self.world.get_blueprint_library()
        # read all valid spawn points
        self.all_default_spawn = self.map.get_spawn_points()

        self.ego_vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.ego_vehicle_bp.set_attribute('color', '0, 0, 0')

        self.num_agents = 5

        self.plan_T = 5.0 # Prediction horizon for MPC 
        self.plan_dt = 0.1 # Sampling time step for MPC

        # simulation parameters ....
        self.sim_T = 50          # Episode length, seconds
        self.sim_dt = 0.1       # simulation time step
        self.max_episode_steps = int(self.sim_T/self.sim_dt)
        ## Planner 
        #self.sigma = 1 # 10
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self):

        # reset the environment

        self.t = 0
        self.initial_u = [0, 0]

        # determine  the start point
        self.spawn_point = self.all_default_spawn[155] 
        #print(self.spawn_point.transform)

        # spawn the ego vehicle
        self.vehicle = self.world.spawn_actor(self.ego_vehicle_bp, self.spawn_point) 
        self.bounding_box = self.vehicle.bounding_box
        self.inter_axle_distance = 2*self.bounding_box.extent.x

        # spawn the moving obstacles (agents)
        self.moving_agents = []
        for i in range(self.num_agents):
            rand_lane_id = random.choice([-1, -2, -3]) # -1,-2,
            rand_s = random.randint(0,150) 
            agent_waypoint = self.map.get_waypoint_xodr(34, rand_lane_id, rand_s)
            spawn_agent_transform = carla.Transform(location=carla.Location(x=agent_waypoint.transform.location.x, \
                                            y=agent_waypoint.transform.location.y, z=agent_waypoint.transform.location.z+0.3),\
                                                            rotation=agent_waypoint.transform.rotation)
            print(rand_lane_id, rand_s)
            moving_agent = spawn_autopilot_agent(self.blueprint_library, self.world, spawn_agent_transform)
            self.moving_agents.append(moving_agent)

        # we need to tick the world once to let the client update the spawn position
        self.world.tick()

        self.destination = self.all_default_spawn[255] 

        self.startpoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
        self.lane_width = self.startpoint.lane_width
        self.vehicle_width = self.vehicle.bounding_box.extent.x * 2 # actually use  length to estimate width with buffer

        self.vehicle_state = get_state_frenet(self.vehicle, self.map)

        self.high_mpc = High_MPC(T=self.plan_T, dt=self.plan_dt, L=self.inter_axle_distance, \
                                 vehicle_width = self.vehicle_width, lane_width = self.lane_width,  init_state=self.vehicle_state)

        # determine and visualize the destination
        self.goal_state = np.array([275, 0, 0, 8]).tolist()
        self.world.debug.draw_point(self.destination.location, size=0.3, color=carla.Color(255,0,0), life_time=300)

    def step(self):
        #
        self.t += self.sim_dt
        done = False

        self.world.tick()
        
        # top view
        self.spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30),
                                                carla.Rotation(pitch=-90)))
        
        # compute the mpc reference
        ref_traj = self.vehicle_state + self.goal_state

        # run  model predictive control
        _act, pred_traj = self.high_mpc.solve(ref_traj)
        control = get_control_input(acceleration=float(_act[0]), steer_angle=float(_act[1]))
        self.vehicle.apply_control(control)
        self.vehicle_state = get_state_frenet(self.vehicle, self.map)
        
        # print current state
        self.curr_waypoint = self.map.get_waypoint(self.vehicle.get_location()) # ,project_to_road=True
        #print('======',self.curr_waypoint.s, '======', self.vehicle_state)
        print('======',self.t, '======', self.curr_waypoint.s)

        # compute the distance toward goal state
        dist2desti = np.linalg.norm(np.array(self.goal_state[:3]) - np.array(self.vehicle_state[:3]))
        if dist2desti < 0.5: #1.25
            print('======== Success, Arrivied at Target Point!')
            done = True      
        elif self.t >= (self.sim_T-self.sim_dt):
            done = True

        '''
        # observation
        self.obs = []

        info = {
            "vehicle_state": self.vehicle_state,
            "act": _act, 
            "pred_vehicle_traj": pred_traj, 
            "current_t": current_t,
            "plan_dt": self.plan_dt
            #"cost": cost
            }
        return self.obs, reward, done, info
        '''
        return done

        
    