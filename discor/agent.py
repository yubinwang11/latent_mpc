import os
import numpy as np
import wandb

from torch.utils.tensorboard import SummaryWriter

from discor.replay_buffer import ReplayBuffer
from discor.utils import RunningMeanStats


class Agent:

    def __init__(self, env, algo, log_dir, device, num_steps=3000000,
                 batch_size=256, memory_size=1000000,
                 update_interval=1, start_steps=10000, log_interval=10,
                 eval_interval=5000, num_eval_episodes=5, seed=0, use_wandb=False, rand_expore=True):

        # Environment.
        self._env = env
        #self._test_env = test_env
        self.use_wandb = use_wandb
        self.rand_explore = rand_expore
        #if use_wandb:
            #self.rand_explore = True

        self._env.seed(seed)
        #self._test_env.seed(2**31-1-seed)

        # Algorithm.
        self._algo = algo
        # Replay buffer with n-step return.
        self._replay_buffer = ReplayBuffer(
            memory_size=memory_size,
            state_shape=self._env.observation_space.shape,#obs.shape[0],,
            action_shape=self._env.action_space.shape,#8,
            gamma=self._algo.gamma, nstep=self._algo.nstep)

        # Directory to log.
        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self._steps = 0
        self._episodes = 0
        self._train_return = RunningMeanStats(log_interval)
        self._writer = SummaryWriter(log_dir=self._summary_dir)
        self._best_eval_score = -np.inf

        self._device = device
        self._num_steps = num_steps
        self._batch_size = batch_size
        self._update_interval = update_interval
        self._start_steps = start_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes

    def run(self):
        while True:
            self.train_episode()
            if self._steps > self._num_steps:
                break

    def train_episode(self):
        self._episodes += 1
        self.episode_return = 0.
        episode_steps = 0

        done = False
        state = self._env.reset()

        while (not done):
            
            #if self.rand_explore:
            if self._start_steps >= self._steps:
                action = self._env.action_space.sample().tolist()
            #else:
                #normalized_state = self.state_normalization(state)
                #normalized_action = self._algo.explore(normalized_state)
                #action = self.anti_normalization(normalized_action) 
                print(f'randomly sampled action: {action}')                
            else:
                normalized_state = self.state_normalization(state)
                normalized_action = self._algo.explore(normalized_state)
                action = self.anti_normalization(normalized_action)
                print(f'neural action: {action}')             
            
            #if self._start_steps > self._steps:
                    #action = self._env.action_space.sample().tolist()
                #else:
                #imitate_flag = True
           # else:
                #imitate_flag = False
            
            #normalized_state = self.state_normalization(state)
            #normalized_action = self._algo.explore(normalized_state)
            #action = self.anti_normalization(normalized_action)

            print(action)
            #if action.type is not list:
               #ref = action.tolist() 
            #else:
            ref = action
            tra_state = np.array(self._env.ego_state) + np.array(ref[0:4])
            tra_state = tra_state.tolist()
            ref_obj = tra_state + ref[4:8]

            # compute the mpc reference
            ref_traj = self._env.ego_state + ref_obj + self._env.goal_state
            # run  model predictive control
            _act, pred_traj = self._env.high_mpc.solve(ref_traj)

            next_state, reward, done, _ = self._env.step(_act, ref)
            self._env.render()

            # Set done=True only when the agent fails, ignoring done signal
            # if the agent reach time horizons.
            if episode_steps + 1 >= self._env.max_time_episode: #_max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            self._replay_buffer.append(
                state, action, reward, next_state, masked_done,
                episode_done=done)

            self._steps += 1
            episode_steps += 1
            self.episode_return += reward
            state = next_state

            if self._steps >= self._start_steps:
                # Update online networks.
                if self._steps % self._update_interval == 0:
                    batch = self._replay_buffer.sample(
                        self._batch_size, self._device)
                    self._algo.update_online_networks(batch, self._writer)

                # Update target networks.
                self._algo.update_target_networks()

                # Evaluate.
                #if self._steps % self._eval_interval == 0:
                    #self.evaluate()
                    #self._algo.save_models(
                        #os.path.join(self._model_dir, 'final'))

            if done:
                aver_reward = self.episode_return / self._env.time_step
                self._env._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
                if self.use_wandb:
                    wandb.log({"episode": self._episodes, "step": self._steps, "return": self.episode_return, "averaged reward": aver_reward})
                                
        # We log running mean of training rewards.
        self._train_return.append(self.episode_return)

        if self._episodes % self._log_interval == 0:
            self._writer.add_scalar(
                'reward/train', self._train_return.get(), self._steps)

        print(f'Episode: {self._episodes:<4}  '
              f'Episode steps: {episode_steps:<4}  '
              f'Return: {self.episode_return:<5.1f}')

    '''
    def evaluate(self):
        total_return = 0.0
        #if self._test_env.is_metaworld:
            #total_success = 0.0

        for _ in range(self._num_eval_episodes):
            state = self._test_env.reset()
            episode_return = 0.0
            done = False
            #if self._test_env.is_metaworld:
                #success = 0.0

            while (not done):
                action = self._algo.exploit(state)
                next_state, reward, done, info = self._test_env.step(action)
                episode_return += reward
                state = next_state

                if self._test_env.is_metaworld and info['success'] > 1e-8:
                    success = 1.0

            total_return += episode_return
            if self._test_env.is_metaworld:
                total_success += success

        mean_return = total_return / self._num_eval_episodes
        if self._test_env.is_metaworld:
            success_rate = total_success / self._num_eval_episodes
            self._writer.add_scalar(
                'reward/success_rate', success_rate, self._steps)

        if mean_return > self._best_eval_score:
            self._best_eval_score = mean_return
            self._algo.save_models(os.path.join(self._model_dir, 'best'))

        self._writer.add_scalar(
            'reward/test', mean_return, self._steps)
        print('-' * 60)
        print(f'Num steps: {self._steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)
        '''

    def __del__(self):
        self._env.close()
        #self._test_env.close()
        #self._writer.close()

    def state_normalization(self, state):
        mean = state.mean(); std = state.std()
        normalized_state = (state - mean)/std

        return normalized_state
    
    def anti_normalization(self, normalized_action):
        action = []
        for dim in range(len(normalized_action)):
            action += [((self._env.act_low[dim]+self._env.act_high[dim])/2 + normalized_action[dim]* (abs(self._env.act_high[dim]-self._env.act_low[dim]))/2).tolist()]
        
        return action