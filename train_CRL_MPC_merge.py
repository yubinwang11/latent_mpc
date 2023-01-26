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

from parameters import *


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_wandb', type=bool, default=True,
                        help="Monitor by wandb")
    parser.add_argument('--episode_num', type=float, default=1000,
                        help="Number of episode")
    parser.add_argument('--save_model_window', type=float, default=32,
                        help="The time gap of saving a model")
    parser.add_argument('--save_model', type=bool, default=True,
                        help="Save the model of nn")
    parser.add_argument('--load_model', type=bool, default=True,
                        help="Load the trained model of nn")
    return parser

def main():

    args = arg_parser().parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    num_episode = args.episode_num

    env_mode = 'challenging'
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

    optimizer = optim.Adam(model.high_policy.parameters(), lr=learning_rate)
    DECAY_STEP = args.save_model_window # 32
    lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_STEP, gamma=0.96)

    if args.load_model:
        model_path = "./" + "models/" + "CRL/"
        print('Loading Model...')
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path + '/checkpoint.pth')
        else:
            checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

    if args.run_wandb:
        wandb.init(
        # set the wandb project where this run will be logged
        project="crl_mpc_training",
        entity="yubinwang",
        # track hyperparameters and run metadata
        config={
        #"learning_rate": learning_rate,
        "exp_decay": env.sigma,
        }
    )

    #env = None

    for episode_i in range(num_episode):
        
        env_mode = 'challenging'

        #if episode_i <= 1000:
            #env_mode = 'easy'

        #elif 1000 < episode_i <= 4000:
           # env_mode = 'medium'
        #else:
            #env_mode = 'hard'
        #if episode_i >= 1000:
            #env_mode = 'medium'
        
        env = MergeEnv(curriculum_mode=env_mode)
        obs=env.reset()

        worker = Worker_Train(env)
        worker_copy_list = [] 
        
        if torch.cuda.is_available():
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            model.to(device)
        else:
            obs = torch.tensor(obs, dtype=torch.float32)

        with autocast():
            high_variable = model.forward(obs)
        
            #scaler = GradScaler()
            z = high_variable

            mean = obs.mean(); std = obs.std()
            high_variable = high_variable*std + mean

            if torch.cuda.is_available():
                high_variable = high_variable.cpu().detach().numpy().tolist()
            else:
                high_variable = high_variable.detach().numpy().tolist()

            for i in range (len(high_variable)):
                worker_copy = copy.deepcopy(worker)
                worker_copy_list.append(worker_copy)

            ep_reward = worker.run_episode(high_variable, args)

            if torch.cuda.is_available():
                #finite_diff_policy_grad = torch.tensor((pertubed_ep_reward - ep_reward)/noise, dtype=torch.float32).to(device)
                finite_diff_policy_grad = torch.tensor(np.zeros(len(high_variable)), dtype=torch.float32).to(device)
            else:
                #finite_diff_policy_grad = torch.tensor((pertubed_ep_reward - ep_reward)/noise, dtype=torch.float32)
                finite_diff_policy_grad = torch.tensor(np.zeros(len(high_variable)), dtype=torch.float32)
            #pertubed_high_variable = np.array(high_variable)
            #pertubed_high_variable = np.zeros(len(high_variable))
            for k in range(len(high_variable)):
                unit_k = np.zeros(len(high_variable))
                unit_k[k] = 1
                #noise_weight = np.random.rand()
                #noise = np.random.randn(len(pertubed_high_variable)) * noise_weight
                noise = np.random.randn() #* noise_weight # 1.5
                noise_vec = unit_k * noise
                pertubed_high_variable = high_variable + noise_vec
                pertubed_high_variable = pertubed_high_variable.tolist()

                pertubed_ep_reward_k = worker_copy_list[k].run_episode(pertubed_high_variable, args) #run_episode(env,goal)
                finite_diff_policy_grad_k = (pertubed_ep_reward_k - ep_reward)/noise
                finite_diff_policy_grad[k] = finite_diff_policy_grad_k
                
            loss = model.compute_loss(-finite_diff_policy_grad, z)
            #loss = z.sum()

        optimizer.zero_grad()

        #torch.autograd.set_detect_anomaly(True)
        #with torch.autograd.detect_anomaly():
            #scaler.scale(loss).backward()
        loss.backward()

        #for param in model.high_policy.parameters():
            #print(param.grad)
            #if param.grad is not None and torch.isnan(param.grad).any():
                #print("nan gradient found")

        #scaler.unscale_(optimizer)
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.high_policy.parameters(), max_norm=10, norm_type=2) # 0.5
        #torch.nn.utils.clip_grad_value_(model.high_policy.parameters(), 0.5)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.high_policy.parameters(), max_norm=10, norm_type=2)

        optimizer.step()
        #scaler.step(optimizer)

        #scale = scaler.get_scale()
        #scaler.update()

        #skip_lr_sched = (scale > scaler.get_scale())
        #if not skip_lr_sched:
        lr_decay.step() #scheduler.step()
        #lr_decay.step()
        best_model = copy.deepcopy(model)

        if args.run_wandb:
            wandb.log({"episode": episode_i, "episode reward": ep_reward, "z_loss": loss, "travese_time": high_variable[-1], "px":high_variable[0], "py": high_variable[1], "grad_norm": grad_norm})
            wandb.watch(model, log='all', log_freq=1)

        if args.save_model:

            if episode_i > 0 and episode_i % args.save_model_window == 0: ##default 100
                #print('Saving model', end='\n')
                model_path = "models/CRL"
                #torch.save(best_model, model_path / 'best_model.pth')
                #path_checkpoint = "./" + model_path + "/best_model.pth"
                #torch.save(best_model, path_checkpoint)
                #torch.save(best_model.state_dict(), path_checkpoint)
                #print('Saved model', end='\n')
                print('Saving model', end='\n')
                checkpoint = {"model": model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "episode": episode_i,
                              "lr_decay": lr_decay.state_dict()}

                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')


    if args.run_wandb:
        wandb.finish()        
    
if __name__ == "__main__":
    main()