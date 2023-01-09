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

from pathlib import Path
import os

import torch
from torch import nn

import wandb

from learning_mpc.merge.merge_env import MergeEnv
from learning_mpc.merge.animation_merge import SimVisual
from networks import DNN
from worker import Worker_Train

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_wandb', type=bool, default=True,
                        help="Monitor by wandb")
    parser.add_argument('--save_model_window', type=float, default=100,
                        help="Play animation")
    parser.add_argument('--save_model', type=bool, default=True,
                        help="Save the model of nn")
    return parser

def main():

    args = arg_parser().parse_args()

    num_episode = 1000
    for episode_i in range(num_episode):
        train(episode_i, args)
    if args.run_wandb:
        wandb.finish()

def train(episode_i, args):


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

    #loss_mse = nn.MSELoss(size_average=False, reduce=True, reduction='mean')

    if args.run_wandb:
        wandb.init(
        # set the wandb project where this run will be logged
        project="crl_mpc_test",
        entity="yubinwang",
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        }
    )

    worker = Worker_Train(env)
    worker_copy = copy.deepcopy(worker)

    optimizer.zero_grad()
    obs = torch.tensor(obs, requires_grad=False, dtype=torch.float32)
    high_variable = model.forward(obs)
    
    #loss = -loss_mse(high_variable, torch.zeros(len(high_variable.detach().numpy())))
    #loss = -loss_mse(high_variable)
    loss = -high_variable.mean()
    loss.backward()
    #print(obs.grad)
    high_variable = high_variable.detach().numpy().tolist()
    #grad2 = obs.grad
    #reward_true = worker.run_episode(high_variable, args)
    ep_reward = worker.run_episode(high_variable, args)
    
    if args.run_wandb:
        wandb.log({"episode reward": ep_reward})

    pertubed_high_variable = np.array(high_variable)
    noise = np.random.randn(len(pertubed_high_variable)) * 0.5 # 1.5
    pertubed_high_variable += noise
    pertubed_high_variable = pertubed_high_variable.tolist()

    pertubed_ep_reward = worker_copy.run_episode(pertubed_high_variable, args) #run_episode(env,goal)
    #print(ep_reward); print(pertubed_ep_reward)
    finite_diff_policy_grad = torch.tensor(pertubed_ep_reward - ep_reward)
    
    #obs.grad *= -finite_diff_policy_grad # gradient asent
    #obs.grad *= finite_diff_policy_grad
    #obs.grad = -obs.grad
    for param in model.high_policy.parameters():
        #print(param.grad.data)
        param.grad.data *= finite_diff_policy_grad
        #print(param.grad.data)
        
    optimizer.step()

    model = copy.deepcopy(model)

    if args.save_model:

        model_dir = Path('./models')

        if episode_i > 0 and episode_i % args.save_model_window == 0: ##default 100
        
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
                    run_num = max(exst_run_nums) + 1 

            curr_run = 'run%i' % run_num
            run_dir = model_dir / curr_run

            os.makedirs(run_dir)
            torch.save(model, run_dir / 'model.pth')
    
if __name__ == "__main__":
    main()