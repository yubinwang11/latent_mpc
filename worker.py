import numpy as np
import time
#import matplotlib.pyplot as plt

import torch

class Worker:

    def __init__(self, env):
        self.env = env
        self.epis_reward = 0
    
    def run_episode(self, high_variable, args):
        
        obs = self.env.obs
        #obs=self.env.reset(self.goal)
        t, n = 0, 0
        t0 = time.time()
        arrived = False
    
        while t < self.env.sim_T:
            t = self.env.sim_dt * n

            #print("current obs: ", obs)
            #print("current high_variable: ", high_variable)
            obs, reward,  done, info = self.env.step(high_variable)
            self.epis_reward += reward
            print("current reward: ", reward)

            #vehicle_pos = np.array(info['quad_s0'][0:3])
            vehicle_pos = np.array(info['vehicle_state'][0:2])
            #print("current pos: ", vehicle_pos)
            if (done):
                arrived = True
                #plt.close()
                break
                
            #t_now = time.time()
            #print(t_now - t0)
            #t0 = time.time()
            #n += 1
            
            if args.visualization:
                update = False
                if t > self.env.sim_T:
                        update = True
                yield [info, t, update]
            
    
        #print("arrvied: ", arrived)
    
        #return self.epis_reward