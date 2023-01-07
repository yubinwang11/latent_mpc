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

from learning_mpc.merge.merge_env import MergeEnv
from learning_mpc.merge.animation_merge import SimVisual
from networks import DNN
from worker import Worker_Eval

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualization', type=bool, default=True,
                        help="Play animation")
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser

def main():

    args = arg_parser().parse_args()

    eval_learningMPC(args)

def eval_learningMPC(args):

    env = MergeEnv()

    obs=env.reset()
    NET_ARCH = [128, 128]
    nn_input_dim = len(obs)
    nn_output_dim = 4 # xy, heading + tra_time
    model = DNN(input_dim=nn_input_dim,
                                output_dim=nn_output_dim,
                                net_arch=NET_ARCH,model_togpu=False)

    worker = Worker_Eval(env)

    obs = torch.tensor(obs, requires_grad=True, dtype=torch.float32)
    #obs = torch.tensor(obs, requires_grad=True)

    high_variable = model.forward(obs)
    high_variable = high_variable.detach().numpy().tolist()

    worker.run_episode(high_variable, args)

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