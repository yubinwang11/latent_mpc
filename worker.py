import numpy as np
import time
#import matplotlib.pyplot as plt

import torch

class Worker_Train:

    def __init__(self, env):
        self.env = env

    def run_episode(self, high_variable, args):
        
        obs = self.env.obs
        #obs=self.env.reset(self.goal)
        t, n = 0, 0
        t0 = time.time()
        arrived = False
        ep_reward = 0
        step_i = 0
        
        while t < self.env.sim_T:
            t = self.env.sim_dt * n
            n += 1
            
            #print("current obs: ", obs)
            #print("current high_variable: ", high_variable)
            obs, reward,  done, info = self.env.step(high_variable, step_i)
            step_i += 1
            ep_reward += reward
            print("current time:", t, "current reward: ", reward, "ep_reward:", ep_reward)
        

            if (done):
                #plt.close()
                print("==========episode of worker is finished==========")
                break
                
            #t_now = time.time()
            #print(t_now - t0)
            #t0 = time.time()          
    
        #print("arrvied: ", arrived)
    
        return ep_reward

class Worker_Eval:

    def __init__(self, env):
        self.env = env

    def run_episode(self, high_variable, args):
        
        obs = self.env.obs
        #obs=self.env.reset(self.goal)
        t, n = 0, 0
        t0 = time.time()
        arrived = False
        ep_reward = 0
        step_i = 0

        while t < self.env.sim_T:
            t = self.env.sim_dt * n
            n += 1
        
            obs, reward,  done, info = self.env.step(high_variable, step_i)
            step_i += 1

            ep_reward += reward
            print("current time:", t, "current reward: ", reward, "ep_reward:", ep_reward)
        
            if (done):
                #plt.close()
                break
                
            #t_now = time.time()
            #print(t_now - t0)
            #t0 = time.time()
            
            #if args.visualization:
            update = False
                #if t > self.env.sim_T:
                        #update = True
            yield [info, t, update, ep_reward]
        
        #print("arrvied: ", arrived)
