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
    parser.add_argument('--use_SE3', type=bool, default=True,
                        help="Baselines")                   
    return parser

def main():

    args = arg_parser().parse_args()
    eval_learningMPC(args)

def eval_learningMPC(args):

    use_learning = True

    eval_mode = 'CRL' # CRL,standardRL or human-expert

    sample_num = 100
    success_sample = 0
    collision_sample = 100

    for i in range(sample_num):
        env_mode = 'hard'
        env = MergeEnv(curriculum_mode=env_mode, eval=False, use_SE3=args.use_SE3)
        obs=env.reset()

        nn_input_dim = len(obs)
        use_gpu = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if (args.use_SE3):
            nn_output_dim=4
        else:
            nn_output_dim = 13

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

        worker = Worker_Record(env)

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

        if (worker.env.collided):
            collision_sample += 1
            print("collision num :", collision_sample, "total num :", i+1)

        if (worker.env.success):
            success_sample += 1
            #print(success_sample, i)
            print("success num :", success_sample, "total num :", i+1)

    success_rate = success_sample /sample_num
    collision_rate = collision_sample /sample_num
    print("collision rate :", collision_rate, "success rate :", success_rate)
    
if __name__ == "__main__":
    main()