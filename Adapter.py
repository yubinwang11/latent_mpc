import numpy as np

'''
Adapters of different Env, Used for better training.
See https://zhuanlan.zhihu.com/p/409553262 for better understanding.
'''


def Reward_adapter(r):
    #if EnvIdex == 0 or EnvIdex == 1:
    if r <= -50: r = -5 # -100 -1
    return r

def Done_adapter(r,done,current_steps):
    if r <= -50: Done = True
    else: Done = False
    return Done

def State_adapter(state):
        mean = state.mean(); std = state.std()
        normalized_state = (state - mean)/std

        return normalized_state

def Action_adapter(act, act_low, act_high):
    #from [-1,1] to [-max,max]
    #return  a*max_action
    action = []
    for dim in range(len(act)):
        action += [((act_low[dim]+act_high[dim])/2 + act[dim]* (abs(act_high[dim]-act_low[dim]))/2).tolist()]
    #print('neural network output is ', act, 'adapted action is ', action)
    return action

def Action_adapter_reverse(act, act_low, act_high):#(act,max_action):
    #from [-max,max] to [-1,1]
    action = []
    for dim in range(len(act)):
        action += [((act[dim]-((act_low[dim]+act_high[dim])/2))/((abs(act_high[dim]-act_low[dim]))/2)).tolist()]
    #print('orginal sampled action is ', act, 'reversed action is ', action)
    return  action
    
    #(act_low[dim]+act_high[dim])/2 = o
    #act[dim] = y 
    #(abs(act_high[dim]-act_low[dim]))/2 = l/2