import copy
import numpy as np
import torch
import torch.nn.functional as F
from surrol.algorithms.TD3.actor import Actor 
from surrol.algorithms.TD3.critic import Critic 
from surrol.algorithms.buffer import ReplayBuffer,her_sampler

import gym
import surrol


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3:
	def __init__(
		self,
		env,
		args,
		writer = None
	):

		# Determine number of goals
		self.num_goals = env.num_goals

		# Space params
		state_dim, action_dim, self.max_action = self._get_space_parameters(env)

		# Create actor 
		self.actor = Actor(state_dim, action_dim, self.max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args['lr'])

		# Create critic
		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args['lr'])

		# Save hyperparameters
		self.discount = args['discount']
		self.tau = args['tau']
		self.policy_noise = args['policy_noise']
		self.noise_clip = args['noise_clip']
		self.policy_freq = args['policy_freq']

		# Flags
		self.total_it = 0

		# Create HER sampler and replay buffer
		sampler = her_sampler(args['replay_k'], self.num_goals, env.compute_reward, env.is_success)
		self.replay_buffer = ReplayBuffer(self.num_goals, env, args['buffer_size'], sampler.sample_her_transitions)

		# Tensorboard agent
		self.writer = writer


	def select_action(self, state, info, goal=None):
		state = self._preprocess_state(state, info, goal)
		return self.actor(state).cpu().data.numpy().flatten()
	

	def train(self, batch_size):
		self.total_it += 1

		# Sample replay buffer 
		batch = self.replay_buffer._sample(batch_size)

		# Preprocess sampled batch
		if self.num_goals > 1:
			state, action, reward, next_state, done = self._preprocess_batch(batch,self.num_goals)
		else:
			state, action, reward, next_state, done = self._preprocess_batch(batch)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_q1, target_q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_q1, target_q2)
			target_Q = reward + (1.0-done) * self.discount * target_Q

		# Get current Q estimates
		current_q1, current_q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_q1, target_Q) + F.mse_loss(current_q2, target_Q)

		# Log critic loss on tensorboard
		self.writer.add_scalar('Train/critic_loss', critic_loss.item(), self.total_it)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			# Log actor loss on tensorboard
			self.writer.add_scalar('Train/actor_loss', actor_loss.item(), self.total_it)
			
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

	def load_network_params(self, params):
		actor_params, critic_params = params
		self.critic.load_state_dict(critic_params)
		self.critic_target = copy.deepcopy(self.critic)
		self.actor.load_state_dict(actor_params)
		self.actor_target = copy.deepcopy(self.actor)

	def get_network_params(self):
		critic = copy.deepcopy(self.critic.state_dict())
		actor = copy.deepcopy(self.actor.state_dict())
		
		return [actor, critic]

	def _preprocess_batch(self,batch,goal=None):

		if goal:
			state = torch.Tensor(np.concatenate((batch['observation'],batch['desired_goal'+str(goal)]),1)).to(device)
		else:
			state = torch.Tensor(np.concatenate((batch['observation'],batch['desired_goal']),1)).to(device)
		action = torch.Tensor(batch['action']).to(device)
		reward = torch.Tensor(batch['reward']).to(device)
		if goal:
			next_state = torch.Tensor(np.concatenate((batch['next_observation'],batch['desired_goal'+str(goal)]),1)).to(device)
		else:
			next_state = torch.Tensor(np.concatenate((batch['next_observation'],batch['desired_goal']),1)).to(device)
		done = torch.Tensor(batch['done']).to(device)

		return state, action, reward, next_state, done

	def _preprocess_state(self, state, info, goal=None):
		if not goal:
			state = np.concatenate((state['observation'],state['desired_goal']),-1)
		else:
			state = np.concatenate((state['observation'],state['desired_goal'+str(goal)]),-1)

		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		return state


	def _get_space_parameters(self, env):
		# Get spaces parameters
		state_dim = 0
		state_dim += env.observation_space.spaces['observation'].shape[0]
		if self.num_goals > 1:
			state_dim += env.observation_space.spaces['desired_goal1'].shape[0]
		else:
			state_dim += env.observation_space.spaces['desired_goal'].shape[0]
		action_dim = env.action_space.shape[0]
		max_action = float(env.action_space.high[0])

		return state_dim, action_dim, max_action


class TD3MultiGoal:
	def __init__(
		self,
		env,
		args,
		writer = None
	):

		# Determine number of goals
		self.num_goals = env.num_goals

		# Space params
		state_dim, action_dim, self.max_action = self._get_space_parameters(env)

		self.actors = []
		self.critics = []
		self.actor_targets = []
		self.critic_targets = []
		self.actor_optimizers = []
		self.critic_optimizers = []

		for i in range(self.num_goals):
			# Create actors
			self.actors.append(Actor(state_dim, action_dim, self.max_action).to(device))
			self.actor_targets.append(copy.deepcopy(self.actors[i]))
			self.actor_optimizers.append(torch.optim.Adam(self.actors[i].parameters(), lr=args['lr']))

			# Create critic
			self.critics.append(Critic(state_dim, action_dim).to(device))
			self.critic_targets.append(copy.deepcopy(self.critics[i]))
			self.critic_optimizers.append(torch.optim.Adam(self.critics[i].parameters(), lr=args['lr']))

		#####################################################################
		################### NeedlePickAndPlace SPECIFIC #####################
		#####################################################################

		# Load pretrained Grasping critic network
		self.critics[1].load_state_dict(torch.load(args['prior_filname'] + "_critic"))
		self.critic_optimizers[1].load_state_dict(torch.load(args['prior_filname'] + "_critic_optimizer"))
		self.critic_targets[1] = copy.deepcopy(self.critics[1])
		self.critics[1].eval()
		self.critic_targets[1].eval()

		# Load pretrained Grasping actor network
		self.actors[1].load_state_dict(torch.load(args['prior_filname'] + "_actor"))
		self.actor_optimizers[1].load_state_dict(torch.load(args['prior_filname'] + "_actor_optimizer"))
		self.actor_targets[1] = copy.deepcopy(self.actors[1])
		self.actors[1].eval()
		self.actor_targets[1].eval()

		#####################################################################
		#####################################################################
		#####################################################################

		# Save hyperparameters
		self.discount = args['discount']
		self.tau = args['tau']
		self.policy_noise = args['policy_noise']
		self.noise_clip = args['noise_clip']
		self.policy_freq = args['policy_freq']

		# Flags
		self.total_it = 0
		self.eps = 0
		self.eps_increment = 1 / (args['max_timesteps'] - args['start_timesteps']) * args['policy_freq']

		# Create HER sampler and replay buffer
		sampler = her_sampler(args['replay_k'], self.num_goals, env.compute_reward, env.is_success)
		self.replay_buffer = ReplayBuffer(self.num_goals, env, args['buffer_size'], sampler.sample_her_transitions)

		# Tensorboard agent
		self.writer = writer


	def select_action(self, state, info, goal=None):
		state = self._preprocess_state(state, info, goal)
		return self.actors[0](state).cpu().data.numpy().flatten()
	

	def train(self, batch_size):
		self.total_it += 1

		# Sample replay buffer 
		batch = self.replay_buffer._sample(batch_size)

		# Preprocess sampled batch
		if self.num_goals > 1:
			state, action, reward, next_state, done = self._preprocess_batch(batch,self.num_goals)
		else:
			state, action, reward, next_state, done = self._preprocess_batch(batch)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_targets[0](next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_q1, target_q2 = self.critic_targets[0](next_state, next_action)
			target_Q = torch.min(target_q1, target_q2)
			target_Q = reward + (1.0-done) * self.discount * target_Q

		# Get current Q estimates
		current_q1, current_q2 = self.critics[0](state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_q1, target_Q) + F.mse_loss(current_q2, target_Q)

		# Log critic loss on tensorboard
		self.writer.add_scalar('Train/critic_loss', critic_loss.item(), self.total_it)

		# Optimize the critic
		self.critic_optimizers[0].zero_grad()
		critic_loss.backward()
		self.critic_optimizers[0].step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute original actor losse
			actor_loss1 = -self.critics[0].Q1(state, self.actors[0](state)).mean()
			self.writer.add_scalar('Train/actor_loss1', actor_loss1.item(), self.total_it // self.policy_freq)

			# Add loss from pretrained Grasping task
			state_alt, _, _, _, _ = self._preprocess_batch(batch,1)
			actor_loss2 = -self.critics[1].Q1(state_alt, self.actors[0](state)).mean()
			self.writer.add_scalar('Train/actor_loss2', actor_loss2.item(), self.total_it // self.policy_freq)

			actor_loss = self.eps * actor_loss1 + (1.0 - self.eps) * actor_loss2

			# Log actor loss on tensorboard
			self.writer.add_scalar('Train/actor_loss', actor_loss.item(), self.total_it // self.policy_freq)

			self.writer.add_scalar('Train/Eps', self.eps, self.total_it // self.policy_freq)
			
			# Optimize the actor 
			self.actor_optimizers[0].zero_grad()
			actor_loss.backward()
			self.actor_optimizers[0].step()

			# Update epsilon
			self.eps += self.eps_increment

			# Update the frozen target models
			for param, target_param in zip(self.critics[0].parameters(), self.critic_targets[0].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actors[0].parameters(), self.actor_targets[0].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critics[0].state_dict(), filename + "_critic")
		torch.save(self.critic_optimizers[0].state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actors[0].state_dict(), filename + "_actor")
		torch.save(self.actor_optimizers[0].state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critics[0].load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizers[0].load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_targets[0] = copy.deepcopy(self.critics[0])

		self.actors[0].load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizers[0].load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_targets[0] = copy.deepcopy(self.actors[0])

	def load_network_params(self, params):
		actor_params, critic_params = params
		self.critics[0].load_state_dict(critic_params)
		self.critic_targets[0] = copy.deepcopy(self.critics[0])
		self.actors[0].load_state_dict(actor_params)
		self.actor_targets[0] = copy.deepcopy(self.actors[0])

	def get_network_params(self):
		critic = copy.deepcopy(self.critics[0].state_dict())
		actor = copy.deepcopy(self.actors[0].state_dict())
		
		return [actor, critic]

	def _preprocess_batch(self,batch,goal=None):

		if goal:
			state = torch.Tensor(np.concatenate((batch['observation'],batch['desired_goal'+str(goal)]),1)).to(device)
		else:
			state = torch.Tensor(np.concatenate((batch['observation'],batch['desired_goal']),1)).to(device)
		action = torch.Tensor(batch['action']).to(device)
		reward = torch.Tensor(batch['reward']).to(device)
		if goal:
			next_state = torch.Tensor(np.concatenate((batch['next_observation'],batch['desired_goal'+str(goal)]),1)).to(device)
		else:
			next_state = torch.Tensor(np.concatenate((batch['next_observation'],batch['desired_goal']),1)).to(device)
		done = torch.Tensor(batch['done']).to(device)

		return state, action, reward, next_state, done

	def _preprocess_state(self, state, info, goal=None):
		if not goal:
			state = np.concatenate((state['observation'],state['desired_goal']),-1)
		else:
			state = np.concatenate((state['observation'],state['desired_goal'+str(goal)]),-1)

		state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		return state


	def _get_space_parameters(self, env):
		# Get spaces parameters
		state_dim = 0
		state_dim += env.observation_space.spaces['observation'].shape[0]
		if self.num_goals > 1:
			state_dim += env.observation_space.spaces['desired_goal1'].shape[0]
		else:
			state_dim += env.observation_space.spaces['desired_goal'].shape[0]
		action_dim = env.action_space.shape[0]
		max_action = float(env.action_space.high[0])

		return state_dim, action_dim, max_action




if __name__ == '__main__':

	env = gym.make('NeedleGrasp-v0')

	algo = TD3(env)

	algo.train()