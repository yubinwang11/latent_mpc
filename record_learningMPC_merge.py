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

from learning_mpc.merge.merge_env import MergeEnv
from learning_mpc.merge.animation_merge import SimVisual
from networks import DNN
from worker import Worker_Record

from parameters import *

def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--visualization', type=bool, default=False,
                        help="Play animation")
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser

def main():

    args = arg_parser().parse_args()
    eval_learningMPC(args)

def eval_learningMPC(args):

    use_learning = True

    eval_mode = 'standardRL' # CRL,standardRL or human-expert

    sample_num = 100
    success_sample = 0

    for i in range(sample_num):
        env_mode = 'hard'
        env = MergeEnv(curriculum_mode=env_mode, eval=False)
        obs=env.reset()

        nn_input_dim = len(obs)
        use_gpu = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = DNN(input_dim=nn_input_dim,
                                    output_dim=nn_output_dim,
                                    net_arch=NET_ARCH,model_togpu=use_gpu,device=device)
        
        if eval_mode == 'CRL':
            model_path = "./" + "models/" + "CRL/"
            print('Loading Model...')
            checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])

        elif eval_mode == 'standardRL':
            model_path = "./" + "models/" + "standardRL/"
            print('Loading Model...')
            checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])\
    
        else:
            use_learning = False
            pass

        worker = Worker_Record(env)

        obs = torch.tensor(obs, dtype=torch.float32)
        mean = obs.mean(); std = obs.std()
        high_variable = model.forward(obs)
        high_variable = high_variable*std + mean

        high_variable = high_variable.detach().numpy().tolist()

        if not (use_learning):
            # decision varaibles are designed by human-expert experience
            high_variable[0] = obs.detach().numpy().tolist()[5]
            high_variable[1] = -env.lane_len/2
            high_variable[2] = 0
            high_variable[3] = 10
            high_variable[4] = 2

        worker.run_episode(high_variable, args)

        if (worker.env.success):
            success_sample += 1
            print(success_sample)
    
    success_rate = success_sample /sample_num
    print(success_rate)
    
if __name__ == "__main__":
    main()