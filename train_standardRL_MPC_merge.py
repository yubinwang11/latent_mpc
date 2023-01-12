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
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.optim as optim

import wandb

from learning_mpc.merge.merge_env import MergeEnv
from learning_mpc.merge.animation_merge import SimVisual
from networks import DNN
from worker import Worker_Train

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_wandb', type=bool, default=True,
                        help="Monitor by wandb")
    parser.add_argument('--episode_num', type=float, default=10000,
                        help="Number of episode")
    parser.add_argument('--save_model_window', type=float, default=32,
                        help="Play animation")
    parser.add_argument('--save_model', type=bool, default=True,
                        help="Save the model of nn")
    return parser

def main():

    args = arg_parser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_episode = args.episode_num

    env_mode = 'general'
    env = MergeEnv(curriculum_mode=env_mode)

    obs=env.reset()
    NET_ARCH = [128, 128, 128, 128]
    nn_input_dim = len(obs)
    nn_output_dim = 4 # xy, heading + tra_time
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    model = DNN(input_dim=nn_input_dim,
                                output_dim=nn_output_dim,
                                net_arch=NET_ARCH,model_togpu=use_gpu,device=device)

    learning_rate = 1e-4
    optimizer = optim.Adam(model.high_policy.parameters(), lr=learning_rate)
    DECAY_STEP = args.save_model_window # 32
    lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=0.96)


    if args.run_wandb:
        wandb.init(
        # set the wandb project where this run will be logged
        project="crl_mpc_test",
        entity="yubinwang",
        # track hyperparameters and run metadata
        config={
        #"learning_rate": learning_rate,
        "exp_decay": env.sigma,
        }
    )

    for episode_i in range(num_episode):
    
        env = MergeEnv()
        obs=env.reset()

        worker = Worker_Train(env)
        worker_copy = copy.deepcopy(worker)
        
        if torch.cuda.is_available():
            obs = torch.tensor(obs, requires_grad=False, dtype=torch.float32).to(device)
            model.to(device)
        else:
            obs = torch.tensor(obs, requires_grad=False, dtype=torch.float32)

        #obs = torch.nn.functional.normalize(obs)
        #with torch.no_grad():

        with autocast():
            high_variable = model.forward(obs)
        
            scaler = GradScaler()
            z = high_variable

            mean = obs.mean(); std = obs.std()
            high_variable = high_variable*std + mean

            if torch.cuda.is_available():
                high_variable = high_variable.cpu().detach().numpy().tolist()
            else:
                high_variable = high_variable.detach().numpy().tolist()

            ep_reward = worker.run_episode(high_variable, args)
        
            pertubed_high_variable = np.array(high_variable)
            noise_weight = np.random.rand()
            noise = np.random.randn(len(pertubed_high_variable)) * noise_weight # 1.5
            pertubed_high_variable += noise
            pertubed_high_variable = pertubed_high_variable.tolist()

            pertubed_ep_reward = worker_copy.run_episode(pertubed_high_variable, args) #run_episode(env,goal)
            #print(ep_reward); print(pertubed_ep_reward)
            if torch.cuda.is_available():
                finite_diff_policy_grad = torch.tensor((pertubed_ep_reward - ep_reward)/noise, dtype=torch.float32).to(device)
            else:
                finite_diff_policy_grad = torch.tensor((pertubed_ep_reward - ep_reward)/noise, dtype=torch.float32)
            
            loss = - model.compute_loss(finite_diff_policy_grad, z)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.high_policy.parameters(), max_norm=10, norm_type=2) # 0.5

        scaler.step(optimizer)
        scaler.update()

        best_model = copy.deepcopy(model)
        lr_decay.step()

        if args.run_wandb:
            wandb.log({"episode reward": ep_reward})
            wandb.log({"z_loss": loss})
            wandb.log({"travese_time": high_variable[-1]})
            wandb.log({"py": high_variable[1]})
            wandb.log({"episode": episode_i})

            wandb.watch(model, log='all', log_freq=1)

        if args.save_model:

            if episode_i > 0 and episode_i % args.save_model_window == 0: ##default 100
                print('Saving model', end='\n')
                model_path = "models/standardRL"
                #torch.save(best_model, model_path / 'best_model.pth')
                path_checkpoint = "./" + model_path + "/best_model.pth"
                torch.save(best_model, path_checkpoint)
                print('Saved model', end='\n')

    if args.run_wandb:
        wandb.finish()        
    
if __name__ == "__main__":
    main()
    