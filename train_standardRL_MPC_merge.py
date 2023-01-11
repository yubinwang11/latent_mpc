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
    #if args.use_crl:
        #curriculum_mode = curriculum_mode_list[0]

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
        #config={
        #"learning_rate": learning_rate,
        #}
    )

    obs = None

    for episode_i in range(num_episode):
    
        env = MergeEnv()
        obs=env.reset()

        worker = Worker_Train(env)
        worker_copy = copy.deepcopy(worker)
        
        #obs = torch.tensor(obs, requires_grad=False, dtype=torch.float32)

        if torch.cuda.is_available():
            obs = torch.tensor(obs, requires_grad=False, dtype=torch.float32).to(device)
            #obs.cuda()
            model.to(device)
            #model.cuda()
        else:
            obs = torch.tensor(obs, requires_grad=False, dtype=torch.float32)

        #with torch.no_grad():
        with autocast():
            high_variable = model.forward(obs)
            scaler = GradScaler()
        #with autocast():
            z = high_variable #loss = -high_variable.sum()
            #loss.requires_grad_(True)

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
                finite_diff_policy_grad = torch.tensor(pertubed_ep_reward - ep_reward).to(device)
            else:
                finite_diff_policy_grad = torch.tensor(pertubed_ep_reward - ep_reward)
            
            loss = - model.compute_loss(finite_diff_policy_grad, z)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        #loss.backward()

        #for param in model.high_policy.parameters():
            #print(param.grad.data)
            #param.grad.data *= finite_diff_policy_grad
            #print(param.grad.data)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.high_policy.parameters(), max_norm=10, norm_type=2)

        #for param in model.high_policy.parameters():
            #print(param.grad.data)

        scaler.step(optimizer)
        scaler.update()

        best_model = copy.deepcopy(model)

        lr_decay.step()

        if args.run_wandb:
            wandb.log({"episode reward": ep_reward})
            wandb.log({"z_loss": loss})
            wandb.watch(model)

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

                print('Saving model', end='\n')
                checkpoint = {"model": best_model.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "episode": episode_i,
                              "lr_decay": lr_decay.state_dict()}

                path_checkpoint = "./" + "models/" + "standardRL" + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                #torch.save(best_model, run_dir / 'model.pth')

    if args.run_wandb:
        wandb.finish()        
    
if __name__ == "__main__":
    main()
    