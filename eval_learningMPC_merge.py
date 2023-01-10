"""
Standard MPC for Autonomous Driving
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import argparse

from pathlib import Path
import os

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

    model_dir = Path('./models')

    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                        model_dir.iterdir() if
                        str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) 

    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run

    model = torch.load(run_dir / 'model.pth')

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