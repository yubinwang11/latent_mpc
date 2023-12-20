#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from high_mpc import High_MPC

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.misc import *


class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.detect_range = params['detect_range']
    self.detector_num = params['detector_num']
    self.detect_angle = params['detect_angle']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.obs_size = int(self.obs_range/self.lidar_bin)


    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # Destination
    self.dests = None

    # action and observation spaces
    self.act_high = np.array([20.0, 15.0, np.pi/2, 20.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32) 
    self.act_low = np.array([-40.0, -15.0, -np.pi/2, -20.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    self.obs_high, self.obs_low = [275.0, 10.0, np.pi/2, 20.0], [0.0, -10, -np.pi/2, -5.0]
    for i in range(self.detector_num):
      self.obs_high.append(50.0)
      self.obs_low.append(0.0)
    self.obs_high = np.array(self.obs_high, dtype=np.float32)
    self.action_space = spaces.Box(
      low=self.act_low, high=self.act_high, dtype=np.float32
      )
    self.observation_space = spaces.Box(low=np.array(self.obs_low), high=np.array(self.obs_high), dtype=np.float32)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(10.0) # 10.0
    self.town_id = 'Town05'
    self.world = client.load_world(self.town_id)
    
    print('Carla server connected!')

    self.map = self.world.get_map()

    self.plan_T = 5.0 # Prediction horizon for MPC 
    self.plan_dt = 0.1 # Sampling time step for MPC

    # simulation parameters ....
    self.sim_T = 50          # Episode length, seconds
    self.sim_dt = 0.1       # simulation time step
    self.max_episode_steps = int(self.sim_T/self.sim_dt)
    
    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)
    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    # create the blueprint library
    self.blueprint_library = self.world.get_blueprint_library()

    # read all valid spawn points
    self.all_default_spawn = self.map.get_spawn_points()

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'])

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Obstacle detector
    self.distance_measurements = []
    self.obstector_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
    self.obstector_bp.set_attribute('debug_linetrace', 'False')
    self.obstector_bp.set_attribute('distance', '50')
    self.obstector_bp.set_attribute('hit_radius', '0.2') #0.5


    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=-6.0, z=2.5))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    #self.camera_bp.set_attribute('fov', '110')
    self.camera_bp.set_attribute('fov', '70')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # Initialize the renderer
    self._init_renderer()

    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T


    self.noise_bound = 5

  def reset(self):
    
    # Clear sensor objects  
    self.collision_sensor = None
    self.camera_sensor = None
    self.detector_list = None

    self._clear_all_actors(['sensor.other.collision', 'sensor.other.obstacle', 'sensor.lidar.ray_cast', \
                           'sensor.camera.rgb', 'vehicle.*'])
    
    # reset time
    self.t = 0
    # reset reward
    self.reward = 0
    # reset done
    self.done = False
    self.arrived = False
    self.out_of_time = False
    self.collided = False
    self.prev_decision_var = None

    self.inter_axle_distance = None

    # Disable sync mode
    self._set_synchronous_mode(True) # True
    
    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    
    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

   
      transform = self.all_default_spawn[155] 
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    ## vehicle param
    self.startpoint = self.map.get_waypoint(self.ego.get_location(), project_to_road=True)
    self.lane_width = self.startpoint.lane_width
    
    self.road_bound_abs = 1.5 * self.lane_width

    self.vehicle_length = self.ego.bounding_box.extent.x * 2
    self.vehicle_width = self.ego.bounding_box.extent.y * 2 # actually use  length to estimate width with buffer
      
    self.inter_axle_distance = 2*self.ego.bounding_box.extent.x

    # determine and visualize the destination
    self.goal_state = np.array([275, 0, 0, 8]).tolist() # 275
    self.destination = self.all_default_spawn[255] 
    self.dests = self.goal_state
    self.road_len = self.goal_state[0]
    
    # spawn the moving obstacles (agents)
    self.moving_agents = []
    
    self.s_list = [15+random.uniform(-self.noise_bound,self.noise_bound), 30+random.uniform(-self.noise_bound,self.noise_bound), \
                    45+random.uniform(-self.noise_bound,self.noise_bound), 60+random.uniform(-self.noise_bound,self.noise_bound), \
                    75+random.uniform(-self.noise_bound,self.noise_bound), 90+random.uniform(-self.noise_bound,self.noise_bound), \
                      105+random.uniform(-self.noise_bound,self.noise_bound),120+random.uniform(-self.noise_bound,self.noise_bound), \
                        135+random.uniform(-self.noise_bound,self.noise_bound) ] #self.s_list = [30, 60, 80, 100, 100, 80, 120]
    
    max_vehicle_distance = 8
    for i in range(len(self.s_list)):
      if i > 0:
        distance_agents = self.s_list[i] - self.s_list[i-1]
        if distance_agents <= max_vehicle_distance:
          distance_refinement = max_vehicle_distance-distance_agents
          self.s_list[i-1] -= distance_refinement / 2 
          self.s_list[i] += distance_refinement / 2 
          
    self.road_id = 34
    self.center_lane_id = -2

    self.num_agents = len(self.s_list)

    for i in range(self.num_agents):
        spawn_lane_id = -random.randint(1,3)
        agent_waypoint = self.map.get_waypoint_xodr(self.road_id, spawn_lane_id, self.s_list[i])
        spawn_agent_transform = carla.Transform(location=carla.Location(x=agent_waypoint.transform.location.x, \
                                    y=agent_waypoint.transform.location.y, z=agent_waypoint.transform.location.z+0.5),\
                                                    rotation=agent_waypoint.transform.rotation)
        moving_agent = self.spawn_autopilot_agent(self.blueprint_library, self.world, spawn_agent_transform)
        self.moving_agents.append(moving_agent)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_hist = []
    
    # Add obstacle dector
    self.distance_measurements = [] #self.det_range
    self.detector_list = []

    for detector_i in range(self.detector_num):
      self.distance_measurements.append(self.detect_range)
      self.obstector_trans = carla.Transform(carla.Location(x=0.0, z=0.5), carla.Rotation(yaw=-self.detect_angle/2+(self.detect_angle/(self.detector_num-1))*detector_i))

      self.detector_list.append(self.world.spawn_actor(self.obstector_bp, self.obstector_trans, attach_to=self.ego))

      self.listen_dector_distance(detector_i)

    # Add camera sensor
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: get_camera_img(data))
    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img = array

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.settings.no_rendering_mode = True
    self.world.apply_settings(self.settings)

    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)

    obs = self._get_obs()

    self.travelled_dist = None

    self.high_mpc = High_MPC(T=self.plan_T, dt=self.plan_dt, L=self.inter_axle_distance, vehicle_length=self.vehicle_length,\
                            vehicle_width = self.vehicle_width, lane_width = self.lane_width,  init_state=self.ego_state)

    return obs
  
  def step(self, action):
        
    # Calculate acceleration and steering
    acc = action[0]
    steer = -action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1) # np.clip(acc/3,0,1) 
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)
    
    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
    #print(act)
    self.ego.apply_control(act)

    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)

    # Update timesteps
    self.t += self.sim_dt
    self.time_step += 1
    self.total_step += 1

    obs = self._get_obs()
    # state information
    info = {
      'ego_state': self.ego_state
    }

    self.done = self._terminal()
    r = self._get_reward()

    return obs,  r, self.done, copy.deepcopy(info) #(obs,  r, self.done, copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self):
    frame = pygame.surfarray.array3d(self.display)
    return frame
  
  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size*2, self.display_size), # * 3
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle
      return True
      
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  def _get_obs(self):
    """Get the observations."""
      ## Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # Roadmap
    if self.pixor:
      roadmap_render_types = ['roadmap']
      self.birdeye_render.render(self.display, roadmap_render_types)
      roadmap = pygame.surfarray.array3d(self.display)
      roadmap = roadmap[0:self.display_size, :, :]
      roadmap = display_to_rgb(roadmap, self.obs_size)
      # Add ego vehicle
      for i in range(self.obs_size):
        for j in range(self.obs_size):
          if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
            roadmap[i, j, :] = birdeye[i, j, :]

    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))
      
    ## Display camera image
    camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
    camera_surface = rgb_to_display_surface(camera, self.display_size)
    self.display.blit(camera_surface, (self.display_size, 0)) # self.display_size * 2
    
    # Display on pygame
    pygame.display.flip()

    self.ego_state = self.get_state_frenet(self.ego, self.map)

    obs = []
    obs += self.ego_state
    obs += self.distance_measurements

    obs = np.array(obs)

    return obs

  def _get_roatation_matrix(self,yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
  
  def _get_reward(self):

    """Calculate the reward."""
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -100

    # reward for steering:
    r_steer = -abs(self.ego.get_control().steer)

    r_speed_ep = 0
    if self.arrived:
      r_speed_ep += self.road_len / self.t 

    r_time = 0 
    if self.out_of_time:
      r_time -= 100

    r_forward = 0 
    current_dist = self.ego_state[0]
    if self.travelled_dist is not None:
      r_forward = current_dist - self.travelled_dist

    # cost for out of road
    r_road = 0
    ego_rotation = self._get_roatation_matrix(self.ego_state[2])
    for corner_id in range(4):
      if corner_id == 0:
          alpha = np.array([self.vehicle_width/2, self.vehicle_length/2]).T
      elif corner_id == 1:
          alpha = np.array([self.vehicle_width/2, -self.vehicle_length/2]).T
      elif corner_id == 2:
          alpha = np.array([-self.vehicle_width/2, -self.vehicle_length/2]).T
      else:
          alpha = np.array([-self.vehicle_width/2, self.vehicle_length/2]).T
      
      corner_pos = self.ego_state[:2] + ego_rotation @ alpha

      if abs(corner_pos[1]) >=self.road_bound_abs:
        dist_road = abs(abs(corner_pos[1]) - self.road_bound_abs)
        r_road = -dist_road
      
    r = r_collision + r_time + r_forward + r_steer + r_speed_ep + r_road
  
    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state

    # If collides
    if len(self.collision_hist)>0: 
      print('end with collision')
      self.collided = True
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      print('end with time')
      self.out_of_time = True
      return True

    if self.dests is not None:
      if self.ego_state[0] >= self.goal_state[0]:
        self.arrived = True
        return True
      
    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        actor.destroy()

  def get_longitudinal_speed(self, vehicle):
    velocity = vehicle.get_velocity()
    forward_vector = vehicle.get_transform().get_forward_vector()
    longitudinal_speed = np.dot(np.array([velocity.x, -velocity.y, velocity.z]), np.array([forward_vector.x,  -forward_vector.y, forward_vector.z]))

    return longitudinal_speed
  
  def get_state_frenet(self, vehicle, map):

    x = map.get_waypoint(vehicle.get_location(), project_to_road=True).s
    centerline_waypoint= map.get_waypoint_xodr(self.road_id,self.center_lane_id, x) # road and lane id
    if centerline_waypoint is None:
      centerline_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
    tangent_vector = centerline_waypoint.transform.get_forward_vector()
    normal_vector = carla.Vector2D(-(-tangent_vector.y), tangent_vector.x)
    norm_normal_vector = np.linalg.norm(np.array([normal_vector.x, normal_vector.y])) 
    normal_vector_normalized = 1 / norm_normal_vector * np.array([normal_vector.x, normal_vector.y]).T
    y_hat = np.array([vehicle.get_location().x-centerline_waypoint.transform.location.x, 
                                    -vehicle.get_location().y-(-centerline_waypoint.transform.location.y)])
    y = np.dot(normal_vector_normalized, y_hat)
    forward_angle = np.arctan2(-tangent_vector.y, tangent_vector.x) * 180/np.pi
    if -180 <= forward_angle < 0:
        forward_angle += 360
    global_yaw = -vehicle.get_transform().rotation.yaw
    if -180 <= global_yaw < 0:
        global_yaw += 360
  
    yaw = (global_yaw-forward_angle)/180 * np.pi
    speed = self.get_longitudinal_speed(vehicle)
    vehicle_state =np.array([x, y, yaw, speed]).tolist()

    return  vehicle_state
  
  def spawn_autopilot_agent(self, blueprint_lib, world, spawn_transform):

    agent_bp = random.choice(blueprint_lib.filter('vehicle.*'))
    agent_bp.set_attribute('role_name', 'autopilot')
    agent = world.spawn_actor(agent_bp, spawn_transform)
    agent.set_autopilot(True)

    return agent
  
  def listen_dector_distance(self, line_i):
    self.detector_list[line_i].listen(lambda distance: get_obstacle_distance(distance, line_i))
    def get_obstacle_distance(info, detector_i):
      if info is not None:
        self.distance_measurements[detector_i] = info.distance
      else:
        self.distance_measurements[detector_i] = self.detect_range 



