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

def get_control_input(acceleration, steer_angle, dead_zone=0.05):

    max_throttle=0.75; max_brake=0.5; max_steering=0.75; KP=1 # 0.1

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

def convert_acc(acc):

    # Convert acceleration to throttle and brake
    if acc > 0:
        throttle = np.clip(acc/3,0,1)
        brake = 0
    else:
        throttle = 0
        brake = np.clip(-acc/8,0,1)

    return throttle, brake
    
def spawn_autopilot_agent(blueprint_lib, world, spawn_transform):

    #agent_bp = random.choice(blueprint_lib.filter('vehicle.*'))
    agent_bp = blueprint_lib.find('vehicle.tesla.model3')
    rand_r, rand_g, rand_b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    agent_bp.set_attribute('color', '{},{},{}'.format(rand_r, rand_g, rand_b))
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

        #self.num_agents = 6

        self.plan_T = 5.0 # Prediction horizon for MPC 
        self.plan_dt = 0.1 # Sampling time step for MPC

        # simulation parameters ....
        self.sim_T = 50          # Episode length, seconds
        self.sim_dt = 0.1       # simulation time step
        self.max_episode_steps = int(self.sim_T/self.sim_dt)

        ## Planner 
        self.sigma = 10 # 10

        # Collision sensor
        self.collision_hist = [] # The collision history
        #self.collision_hist_l = 1 # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    
    def seed(self, seed):
        np.random.seed(seed=seed)
    
    def reset(self):
        
        # reset the environment
        self.collision_sensor = None
        self.reward = 0 

        # Delete surrouding vehicles 
        self._clear_all_actors(['sensor.other.collision', 'vehicle.*'])
        
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
        self.lane_id_list = [-3, -1, -1, -1, -2, -2] #self.lane_id_list = [-3, -1, -1, -1, -2, -2, -2]
        self.s_list = [20, 50, 75, 100, 80, 100] #self.s_list = [30, 60, 80, 100, 100, 80, 120]

        self.num_agents = len(self.lane_id_list)

        for i in range(self.num_agents):
            agent_waypoint = self.map.get_waypoint_xodr(34, self.lane_id_list[i], self.s_list[i])
            spawn_agent_transform = carla.Transform(location=carla.Location(x=agent_waypoint.transform.location.x, \
                                        y=agent_waypoint.transform.location.y, z=agent_waypoint.transform.location.z+0.5),\
                                                        rotation=agent_waypoint.transform.rotation)
            #print(rand_lane_id, rand_s)
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
        self.goal_state = np.array([275, 0, 0, 8]).tolist() # 275
        self.world.debug.draw_point(self.destination.location, size=0.3, color=carla.Color(255,0,0), life_time=300)

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            #if len(self.collision_hist)>self.collision_hist_l:
                #self.collision_hist.pop(0)
        self.collision_hist = []

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
        
        # generate decision variable, now diy for test
        decision_variable = [self.vehicle_state[0]+20, 0, 0, 10, 0.1]
        tra_state = decision_variable[0:4] + [self.t, decision_variable[-1], self.sigma]

        # compute the mpc reference
        ref_traj = self.vehicle_state + tra_state + self.goal_state

        # run  model predictive control
        _act, pred_traj = self.high_mpc.solve(ref_traj)
    
        throttle, brake = convert_acc(_act[0])
        steer = _act[1]
        act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.vehicle.apply_control(act)

        #control = get_control_input(acceleration=float(_act[0]), steer_angle=float(_act[1]))
        #self.vehicle.apply_control(control)

        self.vehicle_state = get_state_frenet(self.vehicle, self.map)
        
        # print current state
        self.curr_waypoint = self.map.get_waypoint(self.vehicle.get_location()) # ,project_to_road=True
        #print('======',self.curr_waypoint.s, '======', self.vehicle_state)
        print('======',self.t, '======', self.curr_waypoint.s)
        #print('===',self.t, '===', self.curr_waypoint.s, '===', pred_traj)
        #self.draw_traj(pred_traj)

        # compute the distance toward goal state
        dist2desti = np.linalg.norm(np.array(self.goal_state[:3]) - np.array(self.vehicle_state[:3]))
        if dist2desti < 1.5: #1.25
            print('======== Success, Arrivied at Target Point!')
            done = True
            self.reward += 100      

        elif self.t >= (self.sim_T-self.sim_dt):
            done = True

        # get the reward

        if done:
            if len(self.collision_hist) > 0:
                #print(self.collision_hist)
                self.reward -= 0.1*self.collision_hist[0]

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

    def draw_traj(self, traj, color=carla.Color(0,0, 255), line_thickness=0.1, duration=60):

        for i in range(len(traj)-1):
            start = carla.Location(x=traj[i][0], y=traj[i][1], z=0)
            end = carla.Location(x=traj[i+1][0], y=traj[i+1][1], z=0)
        
        self.world.debug.draw_line(start, end, line_thickness, color, duration)

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    actor.destroy()
