from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
import pygame

import gym_carla
import carla

import train
import buffer

def main():
	# env = gym.make('BipedalWalker-v2') 
	#env = gym.make('Pendulum-v0')

	# parameters for the gym_carla environment
	params = {
	'number_of_vehicles': 0, # 100
	'number_of_walkers': 0,
	'display_size': 256*2,  # screen size of bird-eye render
	'max_past_step': 1,  # the number of past steps to draw
	'dt': 0.1,  # time interval between two frames
	'discrete': False,  # whether to use discrete control space
	'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
	'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
	'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
	'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
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

	# Set gym-carla environment
	env = gym.make('carla-v0', params=params)
	obs = env.reset()

	MAX_EPISODES = 5000
	MAX_STEPS = 1000
	MAX_BUFFER = 1000000
	MAX_TOTAL_REWARD = 300
	S_DIM = obs.shape[0] #env.observation_space.shape[0]
	A_DIM = 5 # env.action_space.shape[0]
	A_MAX = 50 #env.action_space.high[0]

	print(' State Dimensions :- ', S_DIM) 
	print(' Action Dimensions :- ', A_DIM) 
	print(' Action Max :- ', A_MAX) 

	ram = buffer.MemoryBuffer(MAX_BUFFER)
	trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

	for _ep in range(MAX_EPISODES):

		print('EPISODE :- ', _ep) 
		obs = env.reset()

		for r in range(MAX_STEPS):
			env.render()
			state = np.float32(obs)

			action = trainer.get_exploration_action(state)
			print(action)
			# if _ep%5 == 0:
			# 	# validate every 5th episode
			# 	action = trainer.get_exploitation_action(state)
			# else:
			# 	# get action based on observation, use exploration policy here
			# 	action = trainer.get_exploration_action(state)
			tra_state = action[0:4].tolist() + [env.t, action[-1], 1] # 10 is sigma

			# compute the mpc reference
			ref_traj = env.ego_state + tra_state + env.goal_state
			# run  model predictive control
			_act, pred_traj = env.high_mpc.solve(ref_traj)
		
			new_obs, reward, done, info = env.step(_act)

			# # dont update if this is validation
			# if _ep%50 == 0 or _ep>450:
			# 	continue

			if done:
				new_state = None
				env._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
			else:
				new_state = np.float32(new_obs)
				# push this exp in ram
				ram.add(state, action, reward, new_state)

			obs = new_obs

			# perform optimization
			trainer.optimize()
			if done:
				break

		# check memory consumption and clear memory
		gc.collect()
		# process = psutil.Process(os.getpid())
		# print(process.memory_info().rss)

		#if _ep%100 == 0:
			#trainer.save_models(_ep)

	print('Completed episodes') 

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pygame.quit()
        print(' - Exited by user.')
