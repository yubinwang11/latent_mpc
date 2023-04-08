# -*- coding: utf-8 -*-

import time
import numpy as np
import carla

from high_mpc import High_MPC
from env import Env

def main():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Retrieve the world that is currently running
        world = client.load_world('Town05')

        origin_settings = world.get_settings()

        # set sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1 #0.05
        world.apply_settings(settings)

        # init env
        env = Env(world)

        # reset env
        env.reset()

        while env.t < env.sim_T:

            done = env.step()
            if (done):
                break
    
    finally:
        world.apply_settings(origin_settings)

        env.vehicle.destroy()
        for agent in env.moving_agents:
            agent.destroy()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')