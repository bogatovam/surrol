import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from surrol.algorithms.buffer import ReplayBuffer

import gym
from gym.spaces.dict import Dict
import surrol

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_obs(state):
	if isinstance(state,dict):
		state = np.concatenate((state['observation'],state['desired_goal']),-1)

	return state


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		
		l1 = nn.Linear(state_dim, 512)
		l2 = nn.Linear(512, 1024)
		l3 = nn.Linear(1024, 512)
		l4 = nn.Linear(512, action_dim)

		self.A = nn.Sequential(l1,nn.ReLU(),l2,nn.ReLU(),l3,nn.ReLU(),l4)
		
		self.max_action = max_action
		
	def forward(self, state):
		
		state = preprocess_obs(state)
		out = self.A(state)
		
		return self.max_action * torch.tanh(out)


class Critic(nn.Module):
	def __init__(self,state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		l1 = nn.Linear(state_dim+action_dim,512)
		l2 = nn.Linear(512,1024)
		l3 = nn.Linear(1024,512)
		l4 = nn.Linear(512,1)

		self.C1 = nn.Sequential(l1,nn.ReLU(),l2,nn.ReLU(),l3,nn.ReLU(),l4)

		# Q2 architecture
		l5 = nn.Linear(state_dim+action_dim,512)
		l6 = nn.Linear(512,1024)
		l7 = nn.Linear(1024,512)
		l8 = nn.Linear(512,1)

		self.C2 = nn.Sequential(l5,nn.ReLU(),l6,nn.ReLU(),l7,nn.ReLU(),l8)


	def forward(self,state, action):

		state = preprocess_obs(state)

		input = torch.cat([state,action],1)

		q1 = self.C1(input)
		q2 = self.C2(input)

		return q1,q2



	def Q1(self,state,action):

		state = preprocess_obs(state)

		input = torch.cat([state,action],1)

		q1 = self.C1(input)

		return q1


class TD3:
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		num_goals,
		lr=1e-4,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.num_goals = num_goals

		self.total_it = 0


	def select_action(self, state):
		state = preprocess_obs(state)
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self,buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, done = buffer.sample_goal_specific(batch_size,1)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + (1.0-done) * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)



if __name__ == '__main__':

	env = gym.make('NeedleGrasp-v0')

	algo = TD3(env)

	algo.train()