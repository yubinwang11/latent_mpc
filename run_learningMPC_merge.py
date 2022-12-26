"""
Standard MPC for Autonomous Driving
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import argparse

import torch
from torch import nn

from learning_mpc.merge.mpc_merge import High_MPC
from learning_mpc.merge.merge_env import MergeEnv
from learning_mpc.merge.animation_merge import SimVisual
from networks import DNN
from worker import Worker

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser

def main():

    args = arg_parser().parse_args()

    plan_T = 5.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.1 # Sampling time step for MPC and local planner
    
    init_param = []
    init_param.append(np.array([-28.0, 0.0])) # starting point of the quadrotor
    init_param.append(np.array([5.0]))
    goal = np.array([28, 0.0, 0, 0.0, 0.0, 0.0])
    #goal = np.array([15, 10, 3.14/2, 0.0, 0.0, 0.0])

    initial_state = [-28.0, 0.0, 0, 0, 0, 0]
    initial_u = [0, 0]
    mpc = High_MPC(T=plan_T, dt=plan_dt, init_state=initial_state, init_u=initial_u)
    env = MergeEnv(mpc, plan_T, plan_dt, init_param)

    obs=env.reset(goal)
    NET_ARCH = [128, 128]
    nn_input_dim = len(obs)
    nn_output_dim = 7 # state_dim + tra_time
    model = DNN(input_dim=nn_input_dim,
                                output_dim=nn_output_dim,
                                net_arch=NET_ARCH,model_togpu=False)

    worker = Worker(env, goal)
    worker.run_episode(model) #run_episode(env,goal)
    
    #
    sim_visual = SimVisual(env)
    #
    #run_mpc(env)
    run_frame = partial(worker.run_episode, model)
    ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                  init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)

    
    if args.save_video:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Yubin Wang'), bitrate=1800)
        ani.save("learningMPC_intersection.mp4", writer=writer)

    #plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()