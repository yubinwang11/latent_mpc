import numpy as np
import torch
import gym
import pickle
from SAC import SAC_Agent
from ReplayBuffer import RandomBuffer, device

import gym_carla
import carla
import wandb

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os, shutil
import argparse
from Adapter import *

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--wandb', type=str2bool, default=True, help='Use Wandb to record the training')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--eval', type=str2bool, default=False, help='Evaluate or Not')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=270000, help='which model to load')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--total_steps', type=int, default=int(5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in stpes.')
parser.add_argument('--eval_turn', type=int, default=3, help='Model evaluating times, in episode.') # 3
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
print(opt)

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-5) # 1e-8
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape
    
def evaluate_policy(env, model, render, steps_per_epoch, act_low, act_high, running_state):
    scores = 0
    turns = opt.eval_turn

    for j in range(turns):
        s, done, ep_r = env.reset(), False, 0
        s = running_state(s)
        while not done:
            # Take deterministic actions at test time
            #print('normalized state  is ', s)
            a = model.select_action(s, deterministic=False, with_logprob=False)
            act = Action_adapter(a, act_low, act_high)  # [0,1] to [-max,max]

            #print(act)
            ref = act #.tolist()
            tra_state = np.array(env.ego_state) + np.array(ref[0:4])
            tra_state = tra_state.tolist()
            ref_obj = tra_state + ref[4:8]

            # compute the mpc reference
            ref_traj = env.ego_state + ref_obj + env.goal_state
            # run  model predictive control
            _act, pred_traj = env.high_mpc.solve(ref_traj)

            s_prime, r, done, info = env.step(_act)
            print('under evaluation')

            s_prime = running_state(s_prime)
            # r = Reward_adapter(r, EnvIdex)
            if type(r) == tuple:
                r = np.array(list(r))
            ep_r += r
            s = s_prime
            if render:
                env.render()
        
        # print(ep_r)
        scores += ep_r
    return scores/turns

def main():

    write = opt.write   #Use SummaryWriter to record the training.
    render = opt.render
    eval = opt.eval
    use_wandb = opt.wandb

    # parameters for the gym_carla environment
    params = {
	'number_of_vehicles': 0, # 100
	'number_of_walkers': 0,
	'display_size': 256*2,  # screen size of bird-eye render
	'max_past_step': 1,  # the number of past steps to draw
	'dt': 0.1,  # time interval between two frames
	'discrete': False,  # whether to use discrete control space
	'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
	'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
	#'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
	#'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
	'ego_vehicle_filter': 'vehicle.tesla.model3*',  # filter for defining ego vehicle lincoln
	'port': 2000,  # connection port
	'town': 'Town05',  # which town to simulate
	'task_mode': 'normal',  # mode of the task, [random, normal, roundabout (only for Town03)]
	'max_time_episode': 500,  # maximum timesteps per episode
	'max_waypt': 12,  # maximum number of waypoints
    'detect_range': 50,  # obstacle detection range (meter)
    'detector_num': 19,  # number of obstacle detectiors
    'detect_angle': 180,  # horizontal angle of obstacle detection
	'obs_range': 32,  # observation range (meter)
	'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
	'd_behind': 12,  # distance behind the ego vehicle (meter)
	'out_lane_thres': 2.0,  # threshold for out of lane
	'desired_speed': 8,  # desired speed (m/s)
	'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
	'display_route': False,  # whether to render the desired route
	'pixor_size': 64,  # size of the pixor labels
	'pixor': False,  # whether to output PIXOR observation
    'render': render,
	}
        
    # Create environments.
    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    env_with_Dead = True

    #eval_env = gym.make(EnvName[EnvIdex])
    eval_env = env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    #max_action = float(env.action_space.high[0])
    steps_per_epoch = env.max_episode_steps
    print('Env: CarlaEnv,  state_dim:',state_dim,'  action_dim:',action_dim,
           'max_episode_steps', steps_per_epoch) # '  max_a:',max_action,'  min_a:',env.action_space.low[0],

    #Interaction config:
    start_steps = 5*steps_per_epoch #5*steps_per_epoch #in steps ### ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    update_after = 2*steps_per_epoch #in steps
    update_every = opt.update_every
    total_steps = opt.total_steps
    eval_interval = opt.eval_interval  #in steps
    save_interval = opt.save_interval  #in steps

    #Random seed config:
    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    eval_env.seed(random_seed)
    np.random.seed(random_seed)

    #SummaryWriter config:
    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/SAC_{}'.format('CarlaEnv') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if (use_wandb):
        wandb.init(
                # set the wandb project where this run will be logged
                project="CarlaEnv",
                entity="yubinwang",
                # track hyperparameters and run metadata
                config={
                "algo": "SAC", # discor
                }
            )

    #Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width,opt.net_width),
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size":opt.batch_size,
        "alpha":opt.alpha,
        "adaptive_alpha":opt.adaptive_alpha
    }

    running_state = ZFilter((state_dim,), clip=5.0)
    model = SAC_Agent(**kwargs)

    if not os.path.exists('model'): os.mkdir('model')
    if opt.Loadmodel: 
        model.load(opt.ModelIdex)
        with open("./model/mean_std_{}.txt".format(opt.ModelIdex), 'rb') as saved_mean_std:
                running_state = pickle.load(saved_mean_std)

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_Dead, max_size=int(1e6))

    if eval:
        average_reward = evaluate_policy(eval_env, model, False, steps_per_epoch, env.act_low, env.act_high, running_state) #evaluate_policy(env, model, render, steps_per_epoch, max_action, EnvIdex)
        print('Average Reward:', average_reward)
    else:
        s, done, current_steps = env.reset(), False, 0
        s = running_state(s)
        for t in range(total_steps):

            #s = State_adapter(s)

            current_steps += 1
            '''Interact & trian'''

            if t < start_steps:
                #Random explore for start_steps
                act = env.action_space.sample() #act∈[-max,max]
                act = act.tolist()
                a = Action_adapter_reverse(act,env.act_low, env.act_high) #a∈[-1,1]
            else:
                
                #print('normalized state  is ', s)
                a = model.select_action(s, deterministic=False, with_logprob=False) #a∈[-1,1]
                #a.tolist()
                act = Action_adapter(a, env.act_low, env.act_high) #act∈[-max,max]

            #print(act)
            ref = act #.tolist()
            tra_state = np.array(env.ego_state) + np.array(ref[0:4])
            tra_state = tra_state.tolist()
            ref_obj = tra_state + ref[4:8]

            # compute the mpc reference
            ref_traj = env.ego_state + ref_obj + env.goal_state
            # run  model predictive control
            _act, pred_traj = env.high_mpc.solve(ref_traj)

            s_prime, r, done, info = env.step(_act)
            s_prime = running_state(s_prime)
            if render:
                env.render()

            if type(r) == tuple:
                r = np.array(list(r))

            dead = Done_adapter(r, done, current_steps)
            r = Reward_adapter(r)
            replay_buffer.add(s, a, r, s_prime, dead)
            s = s_prime

            # 50 environment steps company with 50 gradient steps.
            # Stabler than 1 environment step company with 1 gradient step.
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    model.train(replay_buffer)

            '''save model'''
            if (t + 1) % save_interval == 0:
                with open("./model/mean_std_{}.txt".format(t + 1), 'wb') as saved_mean_std:
                    pickle.dump(running_state, saved_mean_std)
                    saved_mean_std.close()

                model.save(t + 1)

            '''record & log'''
            if (t + 1) % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False, steps_per_epoch, env.act_low, env.act_high, running_state)
                if (use_wandb):
                    wandb.log({"step":t+1, "score": score})
                if write:
                    writer.add_scalar('ep_r', score, global_step=t + 1)
                    writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                print('EnvName: CarlaEnv, seed:', random_seed, 'totalsteps:', t+1, 'score:', score)
            if done:
                s, done, current_steps = env.reset(), False, 0
                s = running_state(s)

    #env.close()
    #eval_env.close()

if __name__ == '__main__':
    main()






