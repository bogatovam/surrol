from collections import OrderedDict
import numpy as np
import torch
import gym
import surrol

class ReplayBuffer:

    def __init__(self,
                num_goals,
                env,
                buffer_size):

        self.buffer_size = int(buffer_size)
        self.num_goals = num_goals
        self.size = 0
        self.idx = 0

        self.env = env

        self._buffer = OrderedDict()

        self._init_buffer_storage()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, batch_size):

        ind = np.random.randint(0, self.size, size=batch_size)

        out = dict()

        out['observation'] = torch.FloatTensor(self._buffer['observation'][ind]).to(self.device)
        out['achieved_goal'] = torch.FloatTensor(self._buffer['achieved_goal'][ind]).to(self.device)
        out['action'] = torch.FloatTensor(self._buffer['action'][ind]).to(self.device)
        out['reward'] = torch.FloatTensor(self._buffer['reward'][ind]).to(self.device)
        out['done'] = torch.FloatTensor(self._buffer['done'][ind]).to(self.device)
        out['next_observation'] = torch.FloatTensor(self._buffer['next_observation'][ind]).to(self.device)
        out['next_achieved_goal'] = torch.FloatTensor(self._buffer['next_achieved_goal'][ind]).to(self.device)

        for i in range(self.num_goals):
            out['desired_goal'+str(i+1)] = torch.FloatTensor(self._buffer['desired_goal'+str(i+1)][ind]).to(self.device)
            out['next_desired_goal'+str(i+1)] = torch.FloatTensor(self._buffer['next_desired_goal'+str(i+1)][ind]).to(self.device)
            
        return out

    def sample_goal_specific(self,batch_size,goal=1):

        sample = self.sample(batch_size)

        state = torch.tensor(np.concatenate((sample['observation'],sample['desired_goal'+str(goal)]),-1)).to(self.device)
        action = sample['action'].clone().detach().requires_grad_(True)
        reward = sample['reward'].clone().detach().requires_grad_(True)
        next_state = torch.tensor(np.concatenate((sample['next_observation'],sample['next_desired_goal'+str(goal)]),-1)).to(self.device)
        done = sample['done'].clone().detach().requires_grad_(True)

        return state, action, next_state, reward, done


    def store_transition(self,obs,info,action,next_obs,reward, done, next_info):

        # Ordered lists of desired goals
        desired_goals = list()
        next_desired_goals = list()

        # Iterate through the info dict where the intermediate goals are stored
        for i in range(self.num_goals-1):
            desired_goals.append(info['desired_goal'+str(i+1)])
            next_desired_goals.append(next_info['desired_goal'+str(i+1)])

        # Add the final goal
        desired_goals.append(obs['desired_goal'])
        next_desired_goals.append(next_obs['desired_goal'])

        # Form observation structure
        observation = [obs['observation'],obs['achieved_goal'],desired_goals]
        next_observation = [next_obs['observation'],next_obs['achieved_goal'],next_desired_goals]

        self._add(observation,action,next_observation,reward,done)




    def _init_buffer_storage(self):

        action_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space['observation'].shape[0]
        goal_dim = self.env.observation_space['achieved_goal'].shape[0]

        self._buffer['observation'] = np.zeros((self.buffer_size,obs_dim))
        self._buffer['achieved_goal'] = np.zeros((self.buffer_size,goal_dim))
        self._buffer['action'] = np.zeros((self.buffer_size,action_dim))
        self._buffer['reward'] = np.zeros((self.buffer_size,1))
        self._buffer['done'] = np.zeros((self.buffer_size,1))
        self._buffer['next_observation'] = np.zeros((self.buffer_size,obs_dim))
        self._buffer['next_achieved_goal'] = np.zeros((self.buffer_size,goal_dim))

        for i in range(self.num_goals):
            self._buffer['desired_goal'+str(i+1)] = np.zeros((self.buffer_size,goal_dim))
            self._buffer['next_desired_goal'+str(i+1)] = np.zeros((self.buffer_size,goal_dim))



    def _add(self, obs,  action, next_obs, reward, done):

        observation, achieved_goal, desired_goals = obs
        next_observation, next_achieved_goal, next_desired_goals = next_obs

        self._buffer['observation'][self.idx] = observation
        self._buffer['achieved_goal'][self.idx] = achieved_goal
        self._buffer['action'][self.idx] = action
        self._buffer['reward'][self.idx] = reward
        self._buffer['done'][self.idx] = done
        self._buffer['next_observation'][self.idx] = next_observation
        self._buffer['next_achieved_goal'][self.idx] = next_achieved_goal

        for i in range(self.num_goals):
            self._buffer['desired_goal'+str(i+1)][self.idx] = desired_goals[i]
            self._buffer['next_desired_goal'+str(i+1)][self.idx] = next_desired_goals[i]

        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

   
if __name__ == '__main__':

    num_goals = 2
    buffer_size = 100

    env = gym.make("NeedlePick-v0")

    my_buffer = ReplayBuffer(num_goals,env,buffer_size)

    obs,info = env.reset()

    for i in range(200):

        action = env.action_space.sample()

        next_obs, reward, done, next_info = env.step(action)

        my_buffer.store_transition(obs,info,action,next_obs, reward, done, next_info)

        out = my_buffer.sample(2)

        obs = next_obs




        


