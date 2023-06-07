from __future__ import division

import os
import yaml
import argparse
from datetime import datetime
import torch
import gym

import numpy as np
from torch.autograd import Variable
import pygame
import gym_carla
import carla
import wandb

#from discor.env import make_env
from discor.algorithm import SAC, DisCor
from discor.agent import Agent


def run(args):
    with open(args.config) as f:
       config = yaml.load(f, Loader=yaml.SafeLoader)

    if args.num_steps is not None:
        config['Agent']['num_steps'] = args.num_steps

    # parameters for the gym_carla environment
    params = {
	'use_wandb': True,  # whether to use discrete control space
	'number_of_vehicles': 0, # 100
	'number_of_walkers': 0,
	'display_size': 256*2,  # screen size of bird-eye render
	'max_past_step': 1,  # the number of past steps to draw
	'dt': 0.1,  # time interval between two frames
	'discrete': False,  # whether to use discrete control space
	'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
	'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
	#'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
	#'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
	'ego_vehicle_filter': 'vehicle.tesla.model3*',  # filter for defining ego vehicle lincoln
	'port': 2000,  # connection port
	'town': 'Town05',  # which town to simulate
	'task_mode': 'normal',  # mode of the task, [random, normal, roundabout (only for Town03)]
	'max_time_episode': 500,  # maximum timesteps per episode
	'max_waypt': 12,  # maximum number of waypoints
	'obs_range': 32,  # observation range (meter)
	'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
	'd_behind': 12,  # distance behind the ego vehicle (meter)
	'out_lane_thres': 2.0,  # threshold for out of lane
	'desired_speed': 8,  # desired speed (m/s)
	'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
	'display_route': True,  # whether to render the desired route
	'pixor_size': 64,  # size of the pixor labels
	'pixor': False,  # whether to output PIXOR observation
	}
        
    # Create environments.
    #env = make_env(args.env_id)
    #env = gym.make(args.env_id)
    #test_env = make_env(args.env_id)
    #test_env = gym.make(args.env_id)
    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)

    if params['use_wandb']:
    # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="latent_mpc_test",
            entity="yubinwang",
            # track hyperparameters and run metadata
            config={
            #"learning_rate": 0.02,
            #"architecture": "CNN",
            #"dataset": "CIFAR-100",
            "algo": "{args.algo}", # discor
            }
        )
                
    # Device to use.
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{args.algo}-seed{args.seed}-{time}')

    if args.algo == 'discor':
        # Discor algorithm.
        algo = DisCor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed,
            **config['SAC'], **config['DisCor'])
    elif args.algo == 'sac':
        # SAC algorithm.
        algo = SAC(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=device, seed=args.seed, **config['SAC'])
    else:
        raise Exception('You need to set "--algo sac" or "--algo discor".')

    agent = Agent(
        env=env, algo=algo, log_dir=log_dir,
        device=device, seed=args.seed, use_wandb=params['use_wandb'], **config['Agent'])
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'metaworld.yaml'))
    parser.add_argument('--num_steps', type=int, required=False)
    parser.add_argument('--env_id', type=str, default='Pendulum-v0') # hammer-v1
    parser.add_argument('--algo', choices=['sac', 'discor'], default='discor')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
