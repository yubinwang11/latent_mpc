import numpy as np
import gym
import pygame
import pickle

import gym_carla
import carla
import sys
import traceback
import wandb
import time

from matplotlib import animation
import matplotlib.pyplot as plt
import imageio

from datetime import datetime
import os, shutil
import argparse

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def save_frames_as_gif(frames, run_num=0):

    path='./gif/'
    filename='animation_{}.gif'.format(run_num)

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72) 

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--record', type=str2bool, default=False, help='Record gif or Not')
parser.add_argument('--render', type=str2bool, default=True, help='Render or Not')
parser.add_argument('--plot', type=str2bool, default=True, help='Plot or Not')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--eval_turn', type=int, default=3, help='Model evaluating times, in episode.') # 3
opt = parser.parse_args()
print(opt)


def evaluate_policy(env, render, steps_per_epoch, record=False):
    scores = 0
    turns = opt.eval_turn

    plot = opt.plot
    if plot:
        turns = 1
        wandb.init(
                # set the wandb project where this run will be logged
                project="latent-mpc-plot",
                entity="yubinwang",
                # track hyperparameters and run metadata
                #config={
                #}
            )

    for j in range(turns):
        if record:
            frames = []
        s, done, ep_r = env.reset(), False, 0

        while not done:

            obstacle_state = s

            start_time = time.time()
            # compute the mpc reference
            ref_traj = env.ego_state + obstacle_state + env.goal_state  #

            # run  model predictive control
            _act, pred_traj = env.high_mpc.solve(ref_traj)
            end_time = time.time()

            #print('predicted traj:', pred_traj)
            if plot:
                com_time = end_time - start_time
                wandb.log({"time":env.t, "speed": env.ego_state[-1], "acc":  _act[0],  "steer":  _act[1], "runtime": com_time})

            s_prime, r, done, info = env.step(_act)
            print('current state:', env.ego_state)
            print('computed action:', _act)

            s = s_prime
            
            if type(r) == tuple:
                r = np.array(list(r))
            ep_r += r

            if render:
                env.render()
            if record:
                #frames.append(env.render(mode="rgb_array"))
                #frame = env.render
                frame=np.rot90(pygame.surfarray.array3d(env.display),3)
                frames.append(np.fliplr(frame))
                #frames.append(env.display.copy())

        # print(ep_r)
        scores += ep_r

        if record:
            # save_frames_as_gif(frames, j)
            #frames_pil = imageio.fromarray
           imageio.mimsave('./gif/{}.gif'.format(j), frames, fps=10)


    return scores/turns

def main():

    render = opt.render
    record = opt.record

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
	#'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
	#'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
	'ego_vehicle_filter': 'vehicle.tesla.model3*',  # filter for defining ego vehicle lincoln
	'port': 2000,  # connection port
	'town': 'Town05',  # which town to simulate
	'task_mode': 'normal',  # mode of the task, [random, normal, roundabout (only for Town03)]
	'max_time_episode': 500,  # maximum timesteps per episode
	'max_waypt': 12,  # maximum number of waypoints
    'detect_range': 50,  # obstacle detection range (meter)
    'detector_num': 37,  # number of obstacle detectiors #19
    'detect_angle': 180,  # horizontal angle of obstacle detection
	'obs_range': 32,  # observation range (meter)
	'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
	'd_behind': 12,  # distance behind the ego vehicle (meter)
	'out_lane_thres': 2.0,  # threshold for out of lane
	'desired_speed': 8,  # desired speed (m/s)
	'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
	'display_route': False,  # whether to render the desired route
	'pixor_size': 64,  # size of the pixor labels
	'pixor': False,  # whether to output PIXOR observation
    'render': render,
	}
        
    # Create environments.
    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)

    steps_per_epoch = env.max_episode_steps
    print('Env: CarlaEnv, max_episode_steps', steps_per_epoch) # '  max_a:',max_action,'  min_a:',env.action_space.low[0],

    #Interaction config:

    #Random seed config:
    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    env.seed(random_seed)

    np.random.seed(random_seed)

    average_reward = evaluate_policy(env, render, steps_per_epoch, record=record) 
    print('Average Reward:', average_reward)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 这个是输出错误的具体原因
        print(e) # 输出：division by zero
        print(sys.exc_info()) # 输出：(<class 'ZeroDivisionError'>, ZeroDivisionError('division by zero'), <traceback object at 0x000001A1A7B03380>)
        
        # 以下两步都是输出错误的具体位置，报错行号位置在第几行
        print('\n','>>>' * 20)
        print(traceback.print_exc())
        print('\n','>>>' * 20)
        print(traceback.format_exc())







