"""
Learning MPC for Autonomous Driving
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

from learning_mpc.lane_change.env import Env
from networks import DNN
from trash.worker import Worker_Eval

from parameters import *

def arg_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear_visualization', type=bool, default=False,
                        help="Play animation")
    parser.add_argument('--save_video', type=bool, default=True,
                        help="Save the animation as a video file")
    parser.add_argument('--use_SE3', type=bool, default=False,
                        help="Baselines")  
    return parser

def main():

    args = arg_parser().parse_args()
    eval_learningMPC(args)

def eval_learningMPC(args):
    
    use_learning = True
    eval_mode = 'CRL' # CRL,standardRL or human-expert

    env_mode = 'hard'
    env = Env(curriculum_mode=env_mode, eval=False, use_SE3=args.use_SE3)
    obs=env.reset()

    nn_input_dim = len(obs)
    use_gpu = False

    if (args.use_SE3):
        nn_output_dim=4
    else:
        nn_output_dim = 13

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DNN(input_dim=nn_input_dim,
                                output_dim=nn_output_dim,
                                net_arch=NET_ARCH,model_togpu=use_gpu,device=device)

    if eval_mode == 'CRL':
            if not (args.use_SE3):
                model_path = "./" + "models/augmented/" + "CRL/"
            else:
                model_path = "./" + "models/SE3/" + "CRL/"
            print('Loading Model...')
            checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])

    elif eval_mode == 'standardRL':
        if not (args.use_SE3):
            model_path = "./" + "models/augmented/" + "standardRL/"
        else:
            model_path = "./" + "models/SE3/" + "standardRL/"
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])\

    else:
        use_learning = False
        pass

    worker = Worker_Eval(env)

    obs = torch.tensor(obs, dtype=torch.float32)
    mean = obs.mean(); std = obs.std()
    high_variable = model.forward(obs)
    high_variable = high_variable*std + mean

    high_variable = high_variable.detach().numpy().tolist()

    if not (use_learning):
        # decision varaibles are designed by human-expert experience
        high_variable[0] = obs.detach().numpy().tolist()[5] # px
        high_variable[1] = -env.lane_len/2 # py
        high_variable[2] = 0 # heading
        high_variable[3] = 10 # vx
        high_variable[4] = 0.5 # vy
        high_variable[5] = 0 # omega
        high_variable[6] = 500 # Qx
        high_variable[7] = 500 # Qy
        high_variable[8] = 50 # Qpsi
        high_variable[9] = 10 # Qvx
        high_variable[10] = 10 # Qvy
        high_variable[11] = 10 # Qomega
        high_variable[-1] = 2 # t

    worker.run_episode(high_variable, args)

    if args.clear_visualization:
        sim_visual = SimVisual(worker.env)
    else:
        sim_visual = SimVisual_video(worker.env)
#
    run_frame = partial(worker.run_episode, high_variable, args)
    ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)


    #plt.tight_layout()

    if not args.clear_visualization:
        plt.tight_layout()
        plt.show()
        pass
    #plt.savefig('./1.eps', dpi=300)
    
    if args.save_video:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("trail.mp4", writer=writer)
        #sim_visual.fig.savefig('./1.pdf', dpi=300)
        #sim_visual.fig.savefig('./test.png') 

    
if __name__ == "__main__":
    main()