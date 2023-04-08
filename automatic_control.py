# -*- coding: utf-8 -*-

"""Revised automatic control
"""

import os
import random
import sys
import numpy as np
import carla

from common.high_mpc import High_MPC
from agents.navigation.behavior_agent import BehaviorAgent

def get_longitudinal_speed(vehicle):
    velocity = vehicle.get_velocity()
    forward_vector = vehicle.get_transform().get_forward_vector()
    longitudinal_speed = np.dot(np.array([velocity.x, -velocity.y, velocity.z]), np.array([forward_vector.x,  -forward_vector.y, forward_vector.z]))

    return longitudinal_speed

def get_control_input(acceleration, steer_angle, dead_zone=0.1):
    max_throttle=0.75; max_brake=0.3; max_steering=0.75; KP=0.8 # 0.1
    if acceleration >= dead_zone:
        throttle = min(max_throttle, KP*acceleration)
        brake = 0
    elif acceleration <= -dead_zone:
        throttle = 0 
        brake = min(max_brake, -KP*acceleration)

    desired_steer_angle = np.clip(steer_angle/max_steering, -1, 1)

    control = carla.VehicleControl(throttle=throttle, brake=brake, steer=desired_steer_angle, hand_brake=False)

    return control

def get_state_frenet(vehicle, map):

    x = map.get_waypoint(vehicle.get_location(), project_to_road=True).s
    #centerline_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
    centerline_waypoint= map.get_waypoint_xodr(34, -2,x) # road and lane id
    tangent_vector = centerline_waypoint.transform.get_forward_vector()
    normal_vector = carla.Vector3D(-(-tangent_vector.y), tangent_vector.x, 0)
    normal_vector_normalized = np.array([normal_vector.x, -normal_vector.y]) /  np.linalg.norm(np.array([normal_vector.x, -normal_vector.y]))
    y_hat = np.array([vehicle.get_location().x-centerline_waypoint.transform.location.x, 
                                    -vehicle.get_location().y-(-centerline_waypoint.transform.location.y)])
    y = np.dot(normal_vector_normalized, y_hat)
    forward_angle = np.arctan2(-tangent_vector.y, tangent_vector.x) * 180/np.pi
    yaw = (forward_angle - (-vehicle.get_transform().rotation.yaw))/180 * np.pi
    speed = get_longitudinal_speed(vehicle)
    #vehicle_state =np.array([vehicle_x, vehicle_y, vehicle_yaw,  vehicle.get_velocity().x, -vehicle.get_velocity().y, -vehicle.get_angular_velocity().z])
    vehicle_state =np.array([x, y, yaw, speed]).tolist()

    return  vehicle_state

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        # Retrieve the world that is currently running
        world = client.load_world('Town05')

        origin_settings = world.get_settings()

        # set sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1 #0.05
        world.apply_settings(settings)

        map = world.get_map()

        blueprint_library = world.get_blueprint_library()

        # read all valid spawn points
        all_default_spawn = world.get_map().get_spawn_points()
        # determine  the start point
        spawn_point = all_default_spawn[155] 
        # create the blueprint library
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point) 
        bounding_box = vehicle.bounding_box
        inter_axle_distance = 2*bounding_box.extent.x

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        # create the behavior agent
        agent = BehaviorAgent(vehicle, behavior='normal') # normal

        destination = all_default_spawn[255] 

        # generate the route
        agent.set_destination(destination.location, spawn_point.location)

        plan_T = 5.0 # Prediction horizon for MPC and local planner
        plan_dt = 0.1 # Sampling time step for MPC and local planner

        curr_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
        lane_width = curr_waypoint.lane_width
        vehicle_width = vehicle.bounding_box.extent.x * 2

        vehicle_state = get_state_frenet(vehicle, map)
        high_mpc = High_MPC(T=plan_T, dt=plan_dt, L=inter_axle_distance, vehicle_width = vehicle_width, lane_width = lane_width,  init_state=vehicle_state)

        goal_state = np.array([270, 0, 0, 10]).tolist()
        '''
        # generate the centerline on road
        total_length=271
        incre_dist = 0.5
        centerline_points = []
        curr_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)
        #curr_travel_length = curr_waypoint.
        while curr_waypoint:
            centerline_points.append(curr_waypoint)
            curr_waypoint = curr_waypoint.next(incre_dist)[0] if curr_waypoint.next(incre_dist) else None
            if len(centerline_points) > total_length/incre_dist: #830: 
                break
        #print(centerline_points)
        '''

        # visualize all centerline_point
       # for waypoint in centerline_points:
            #world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(0,255,0), life_time=300)
        world.debug.draw_point(destination.transform.location, size=0.5, color=carla.Color(255,0,0), life_time=300)
        #vehicle_state = np.zeros(6)

        while True:
            agent._update_information() #agent._update_information(vehicle)
            vehicle_state = get_state_frenet(vehicle, map)
            world.tick()
            
            if len(agent._local_planner._waypoints_queue)<1:
                print('======== Success, Arrivied at Target Point!')
                break
                
            # top view
            spectator = world.get_spectator()
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                    carla.Rotation(pitch=-90)))

            speed_limit = vehicle.get_speed_limit()
            agent.get_local_planner().set_speed(speed_limit)

            #control = agent.run_step(debug=True)
            #vehicle.apply_control(control)
            #print(location) 
            ego_waypoint = map.get_waypoint(vehicle.get_location()) # ,project_to_road=True
            #print(ego_waypoint.road_id, ego_waypoint.lane_id)
            s = ego_waypoint.s
            #print(s, ego_waypoint.transform.rotation.yaw)
            print(s, vehicle_state)

            ref_traj = vehicle_state + goal_state

            # - -----------------------------------------------------------
            # run  model predictive control
            _act, pred_traj = high_mpc.solve(ref_traj)
            control = get_control_input(acceleration=float(_act[0]), steer_angle=float(_act[1]))
            vehicle.apply_control(control)

            dist2desti = np.linalg.norm(np.array(goal_state) - np.array(vehicle_state))
            if dist2desti < 5: #1.25
                break

    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')