from collections import OrderedDict
import numpy as np
import torch
import gym
import surrol

from surrol.algorithms.buffer.her import her_sampler

class ReplayBuffer:

    def __init__(self,
                num_goals,
                env,
                buffer_size,
                sample_function,
                prioritised=True):

        # Prioratise parameters
        self.prioritised = prioritised

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_function = sample_function
        self.env = env

        # Max number of transitions
        self.buffer_size = int(buffer_size)

        # Maximum number of steps in an episode
        self.T = env._max_episode_steps

        # Max number of trajectories
        self.max_size = self.buffer_size // self.T

        # Number of goals to save in the buffer
        self.num_goals = num_goals

        # Buffer size counts 
        self.num_transitions_stored = 0
        self.num_trajectories_stored = 0

        # Storing index on the ring buffer
        self.idx_ptr = 0

        # Init buffer structure
        self._buffer = OrderedDict()
        self._init_buffer_storage()

    def is_full(self):
        return self.num_trajectories_stored == self.max_size

    def is_empty(self):
        return self.num_trajectories_stored == 0

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        self._buffer["priorities"][idxs] = priorities.reshape(-1,1)

    def _sample(self, batch_size):

        # Create empty batch structure
        temp_buffer = {'info': dict()}

        # Copy the allocated replay buffer data
        for key in self._buffer.keys():
            if key == 'info':
                for info_key in self._buffer['info'].keys():
                    temp_buffer['info'][info_key] = self._buffer[key][info_key][:self.num_trajectories_stored]
            else:
                temp_buffer[key] = self._buffer[key][:self.num_trajectories_stored]

        # Create next values by shifting current values by one
        temp_buffer['next_observation'] = temp_buffer['observation'][:,1:,:]
        temp_buffer['next_achieved_goal'] = temp_buffer['achieved_goal'][:,1:,:]
        
        next_infos = {}
        for key, val in temp_buffer['info'].items():
            next_infos[key] = val[:,1:,:]
        temp_buffer['next_info'] = next_infos

        # Sample from temporary buffer using sampling function (i.e. HER)
        batch = self.sample_function(temp_buffer, batch_size)
            
        return batch


    def _init_buffer_storage(self):

        # Dimensions for action, state and goal spaces
        obs_dim, goal_dim, action_dim = self.env.get_space_dims()

        # Initialise empty arrays for obs, achieved_goal and actions
        self._buffer['observation'] = np.zeros((self.max_size, self.T + 1, obs_dim))
        self._buffer['achieved_goal'] = np.zeros((self.max_size, self.T + 1, goal_dim))
        self._buffer['action'] = np.zeros((self.max_size, self.T, action_dim))

        if self.prioritised:
            self._buffer['priorities'] = np.zeros((self.max_size, 1))

        if self.num_goals > 1:
            # Initialise storage for multiple goals
            for i in range(self.num_goals):
                self._buffer['desired_goal'+str(i+1)] = np.zeros((self.max_size,self.T + 1, goal_dim))
        else:
            # Initialise storage for a single goal
            self._buffer['desired_goal'] = np.zeros((self.max_size,self.T + 1, goal_dim))

        self._buffer['info'] = dict()
        for key, val in self.env.env.info_space.items():
            self._buffer['info'][key] = np.zeros((self.max_size, self.T + 1, val.shape[0]))



    def _store_transitions(self, episode):

        # Unpack episode batch, shape [batch_size,:]
        observations, achieved_goals, desired_goals, actions, infos = episode
        batch_size = observations.shape[0]

        # Get idx defining where to store on the ring buffer
        idx = self._get_storage_idx(batch_size)

        # Reset the buffer at the chosen indexes
        self._reset_buffer_at_idx(idx)

        # Store the data
        self._buffer['observation'][idx] = observations
        self._buffer['achieved_goal'][idx] = achieved_goals
        self._buffer['action'][idx] = actions

        # Add maximum priority to the new samples
        if self.prioritised:
            priorities = np.ones((batch_size)) if self.is_empty() else np.ones((batch_size)) * self._buffer["priorities"].max()
            self._buffer['priorities'][idx] = priorities
            

        if self.num_goals > 1:
            for i in range(self.num_goals):
                self._buffer['desired_goal'+str(i+1)][idx] = desired_goals[:,:,i,:]
        else:
            self._buffer['desired_goal'][idx] = desired_goals[:,:,0,:]

        for key, val in infos.items():
            self._buffer['info'][key][idx] = val

        # Update total count of transitions stored
        self.num_transitions_stored = min(batch_size * self.T + self.num_transitions_stored, 
                                            self.max_size * self.T)

        # Update total count of trejctories stored
        self.num_trajectories_stored = min(batch_size + self.num_trajectories_stored, self.max_size)

    def _reset_buffer_at_idx(self,idx):

        # Reset the static elements
        self._buffer['observation'][idx] = np.zeros_like(self._buffer['observation'][idx])
        self._buffer['achieved_goal'][idx] = np.zeros_like(self._buffer['achieved_goal'][idx])
        self._buffer['action'][idx] = np.zeros_like(self._buffer['action'][idx])

        # Reset the goal elements
        if self.num_goals > 1:
            for i in range(self.num_goals):
                self._buffer['desired_goal'+str(i+1)][idx] = np.zeros_like(self._buffer['desired_goal'+str(i+1)][idx])
        else:
            self._buffer['desired_goal'][idx] = np.zeros_like(self._buffer['desired_goal'][idx])

        # Reset the info elements
        for key in self._buffer['info']:
            self._buffer['info'][key][idx] = np.zeros_like(self._buffer['info'][key][idx])


    def _get_storage_idx(self, increment=None):

        # If increment not specified assume one trajectory is added
        increment = increment or 1

        # If buffer can fit the increment, store at the end of the buffer
        if self.idx_ptr + increment <= self.max_size:
            idx = np.arange(self.idx_ptr, self.idx_ptr + increment)
            self.idx_ptr += increment

        # Cycle the storage
        else:
            overhead = increment - (self.max_size - self.idx_ptr)
            idx_tail = np.arange(self.idx_ptr, self.max_size)
            idx_head = np.arange(0,overhead)
            idx = np.concatenate([idx_tail,idx_head])
            self.idx_ptr = overhead

        # If one trajectory added, make idx scalar
        #if increment == 1:
            #idx = idx[0]

        return idx


if __name__ == '__main__':

    num_goals = 2
    batch_size = 128
    replay_k = 4
    buffer_size = 101

    env = gym.make("NeedleReach-v0")

    num_goals = env.num_goals

    sampler = her_sampler(replay_k, num_goals, env.compute_reward, env.is_success)
    my_buffer = ReplayBuffer(num_goals,env,buffer_size,sampler.sample_her_transitions)

    for _ in range(3):

        state = env.reset()

        observations = np.zeros((env._max_episode_steps + 1, env.observation_space.spaces['observation'].shape[0]))
        achieved_goals = np.zeros((env._max_episode_steps + 1, env.observation_space.spaces['achieved_goal'].shape[0]))
        desired_goals = np.zeros((env._max_episode_steps + 1, num_goals, env.observation_space.spaces['desired_goal'].shape[0]))
        actions = np.zeros((env._max_episode_steps, env.action_space.shape[0]))

        observations[0] = state['observation']
        achieved_goals[0] = state['achieved_goal']
        desired_goals[0][0] = state['desired_goal']


        for i in range(50):

            action = env.action_space.sample()

            actions[i] = action

            next_state, reward, done, next_info = env.step(action)

            observations[i+1] = next_state['observation']
            achieved_goals[i+1] = next_state['achieved_goal']
            desired_goals[i+1][0] = next_state['desired_goal']

            state = next_state


        episode = [np.expand_dims(observations,0), 
                np.expand_dims(achieved_goals,0), 
                np.expand_dims(desired_goals,0), 
                np.expand_dims(actions,0)]

        my_buffer._store_transitions(episode)

    my_buffer._sample(batch_size)




        


