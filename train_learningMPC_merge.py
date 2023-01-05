"""
Standard MPC for Autonomous Driving
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import argparse
import copy

import torch
from torch import nn

from learning_mpc.merge.merge_env import MergeEnv
from learning_mpc.merge.animation_merge import SimVisual
from networks import DNN
from worker import Worker

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualization', type=bool, default=True,
                        help="Play animation")
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser

def main():

    args = arg_parser().parse_args()

    num_episode = 100
    for i in range(num_episode):
        train(args)

def train(args):

    env = MergeEnv()

    obs=env.reset()
    NET_ARCH = [128, 128]
    nn_input_dim = len(obs)
    nn_output_dim = 4 # xy, heading + tra_time
    model = DNN(input_dim=nn_input_dim,
                                output_dim=nn_output_dim,
                                net_arch=NET_ARCH,model_togpu=False)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.high_policy.parameters(), lr=learning_rate)

    loss_mse = nn.MSELoss(size_average=False, reduce=True, reduction='mean')

    worker = Worker(env)
    worker_copy = copy.deepcopy(worker)

    optimizer.zero_grad()
    obs = torch.tensor(obs, requires_grad=True)
    high_variable = model.forward(obs)
    
    print(obs.grad)
    loss_mse(high_variable, torch.zeros(len(high_variable.detach().numpy()))).backward()
    print(obs.grad)
    high_variable = high_variable.detach().numpy().tolist()
    grad2 = obs.grad
    #reward_true = worker.run_episode(high_variable, args)
    worker.run_episode(high_variable, args)

    '''
    pertubed_high_variable = np.array(high_variable)
    noise = 0.3*np.random.randn(len(pertubed_high_variable))
    pertubed_high_variable += noise
    pertubed_high_variable = pertubed_high_variable.tolist()

    reward_bsl = worker_copy.run_episode(pertubed_high_variable, args) #run_episode(env,goal)
    print(reward_true); print(reward_bsl)
    grad1 = torch.tensor(reward_bsl - reward_true)
    total_grad = grad1 * grad2
    
    optimizer.step(total_grad)
    print(total_grad)
    '''
    if args.visualization:
        sim_visual = SimVisual(env)
        #
        run_frame = partial(worker.run_episode, high_variable, args)
        ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                    init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)
        
        plt.tight_layout()
        plt.show()
    
    if args.save_video:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Yubin Wang'), bitrate=1800)
        ani.save("learningMPC_merge.mp4", writer=writer)

    
if __name__ == "__main__":
    main()