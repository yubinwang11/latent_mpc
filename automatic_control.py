# -*- coding: utf-8 -*-

"""Revised automatic control
"""

import os
import random
import sys
import numpy as np
import carla

from agents.navigation.behavior_agent import BehaviorAgent

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
        # randomly choose one as the start point
        spawn_point = all_default_spawn[155] 
        #spawn_point = carla.Transform(carla.Location(x=45.0, y=-205.0, z=0.000000),carla.Rotation(pitch=0.000000, yaw=-90.669098, roll=0.000000))
        # create the blueprint library
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # spawn the vehicle
        vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point) # 0.267604261636754 

        # we need to tick the world once to let the client update the spawn position
        world.tick()

        # create the behavior agent
        agent = BehaviorAgent(vehicle, behavior='normal') # normal

        destination = all_default_spawn[255] 

        # generate the route
        agent.set_destination(destination.location, spawn_point.location)

        # generate the centerline on road
        total_length=271
        incre_dist = 0.5
        centerline_points = []
        curr_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)

        vehicle_state = np.zeros(6)
        while curr_waypoint:
            centerline_points.append(curr_waypoint)
            curr_waypoint = curr_waypoint.next(incre_dist)[0] if curr_waypoint.next(incre_dist) else None
            if len(centerline_points) > total_length/incre_dist: #830: 
                break
        #print(centerline_points)

        # visualize all centerline_point

        for waypoint in centerline_points:
            world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(0,255,0), life_time=300)

        while True:
            agent._update_information() #agent._update_information(vehicle)

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

            control = agent.run_step(debug=True)
            vehicle.apply_control(control)

            location = vehicle.get_location()
            #print(location) 
            ego_waypoint = map.get_waypoint(vehicle.get_location()) # ,project_to_road=True
            s = ego_waypoint.s
            #print(s, ego_waypoint.transform)
            print(s)

            ego_control = carla.VehicleControl()
            ego_control.throttle = 0.0
            ego_control.steer = 0.0
            ego_control.brake = 0
            ego_control.hand_brake = False

    finally:
        world.apply_settings(origin_settings)
        vehicle.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')