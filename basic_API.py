"""
In this script, we are going to learn how to spawn a vehicle on the road and make it autopilot.
At the same time, we will collect camera and lidar data from it.
"""

import carla
import os
import random


def main():
    actor_list = []
    sensor_list = []

    try:
        # First of all, we need to create the client that will send the requests, assume port is 2000
        client=carla.Client(host='127.0.0.1', port=2000)
        client.set_timeout(200)

        # Retrieve the world that is currently running
        # world = client.get_world()
        world = client.load_world('Town07') # you can also retrive another world by specifically defining
        blueprint_library = world.get_blueprint_library()
        # Set weather for your world
        #weather = carla.WeatherParameters(cloudiness=10.0,
                                          #precipitation=10.0,
                                          #fog_density=10.0)
        #world.set_weather(weather)

        # create the ego vehicle
        ego_vehicle_bp = blueprint_library.find('vehicle.mercedes.coupe')
        # black color
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # get a random valid occupation in the world
        transform = random.choice(world.get_map().get_spawn_points())
        # spawn the vehilce
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
        # set the vehicle autopilot mode
        ego_vehicle.set_autopilot(True)

        # collect all actors to destroy when we quit the script
        actor_list.append(ego_vehicle)

        while True:
            # set the sectator to follow the ego vehicle
            spectator = world.get_spectator()
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
                                                    carla.Rotation(pitch=-90)))
            
            loc = ego_vehicle.get_location()
            print(loc)

    finally:
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')