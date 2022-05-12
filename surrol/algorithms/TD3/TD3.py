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
		# Use prioritised sampling
		self.prioritised = args['buffer']['prioritised']

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
		if self.prioritised:
			# Annealing beta for importance sampling
			self.beta = args['buffer']['beta']
			self.beta_increment = (1.0 - self.beta) / (args['max_timesteps'] - args['start_timesteps'])

		# Create HER sampler and replay buffer
		sampler = her_sampler(args, self.num_goals, env.compute_reward, env.is_success)
		self.replay_buffer = ReplayBuffer(self.num_goals, env, args['buffer']['buffer_size'], sampler.sample_her_transitions, args['buffer']['prioritised'])

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
			state, action, reward, next_state, done, info = self._preprocess_batch(batch, self.num_goals)
		else:
			state, action, reward, next_state, done, info = self._preprocess_batch(batch)

		# Get target Q using 2 target critics
		target_Q = self._get_target_Q(action, next_state, reward, done)

		# Get current Q estimates
		current_q1, current_q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = self._compute_critic_loss(current_q1, current_q2, target_Q, info)

		# Log critic loss on tensorboard
		self.writer.add_scalar('Train/critic_loss', critic_loss.item(), self.total_it)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Train callback
		self.train_callback()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = self._compute_actor_loss(state, batch)

			# Log actor loss on tensorboard
			self.writer.add_scalar('Train/actor_loss', actor_loss.item(), self.total_it)
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			self._polyak_update()


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

	def eval(self):
		self.critic.eval()
		self.actor.eval()

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

	def _get_target_Q(self, action, next_state, reward, done):

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
		
		return target_Q

	def _compute_critic_loss(self, current_q1, current_q2, target_Q, info):

		# Compute critic loss
		if self.prioritised:
			TD_q1 = current_q1 - target_Q
			TD_q2 = current_q2 - target_Q
			TD_errors = (abs(TD_q1) + abs(TD_q2)) / 2
			self.replay_buffer.update_priorities(info['episode_idxs'], TD_errors.squeeze().detach().cpu().numpy())
			err_q1 = torch.mean((TD_q1 * info['weights'])**2)
			err_q2 = torch.mean((TD_q2 * info['weights'])**2)
			critic_loss = err_q1 + err_q2

		else:
			critic_loss = F.mse_loss(current_q1, target_Q) + F.mse_loss(current_q2, target_Q)

		return critic_loss

	def _compute_actor_loss(self, state, batch):

		actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
		return actor_loss

	def train_callback(self):

		if self.prioritised:

			# Log Beta
			self.writer.add_scalar('Train/Beta', self.beta, self.total_it)

			# Update Beta
			self.beta += self.beta_increment


	def _polyak_update(self):

		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
	

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

		info = {}
		if self.prioritised:
			info['episode_idxs'] = batch['episode_idxs']
			info['weights'] = torch.Tensor(batch['weights']).to(device)

		return state, action, reward, next_state, done, info

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


class TD3MultiGoal(TD3):
	def __init__(
		self,
		env,
		args,
		writer = None
	):
		super(TD3MultiGoal, self).__init__(env,args,writer)

		#####################################################################
		################### NeedlePickAndPlace SPECIFIC #####################
		#####################################################################

		# Load pretrained Grasping critic network
		self.grasp_critic = copy.deepcopy(self.critic)
		self.grasp_critic.load_state_dict(torch.load(args['prior_filname'] + "_critic"))
		self.grasp_critic.eval()

		#####################################################################
		#####################################################################
		#####################################################################

		# Eps for critic transfer smootheness
		self.eps = 0
		self.eps_increment = 1 / (args['max_timesteps'] - args['start_timesteps'] - args['train_after_decay'])
	

	def _compute_actor_loss(self, state, batch):

		# Compute original actor losse
		actor_loss1 = -self.critic.Q1(state, self.actor(state)).mean()
		self.writer.add_scalar('Train/actor_loss1', actor_loss1.item(), self.total_it // self.policy_freq)

		# Add loss from pretrained Grasping task
		state_alt, _, _, _, _, _ = self._preprocess_batch(batch,1)
		actor_loss2 = -self.grasp_critic.Q1(state_alt, self.actor(state)).mean()
		self.writer.add_scalar('Train/actor_loss2', actor_loss2.item(), self.total_it // self.policy_freq)

		actor_loss = self.eps * actor_loss1 + (1.0 - self.eps) * actor_loss2

		return actor_loss

	def train_callback(self):
		super().train_callback()

		# Log epsilon
		self.writer.add_scalar('Train/Eps', self.eps, self.total_it)

		# Update epsilon
		self.eps = min(1, self.eps + self.eps_increment)



if __name__ == '__main__':

	env = gym.make('NeedleGrasp-v0')

	algo = TD3(env)

	algo.train()