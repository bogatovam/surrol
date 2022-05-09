import numpy as np
import gym
from multiprocessing import Process, Queue

def run_eval_policy(net_params, net_name, args, env_name, seed, queue, eval_episodes):
		eval_env = gym.make(env_name)
		eval_env.seed(seed + 100)
		goal = eval_env.num_goals if eval_env.num_goals > 1 else None

		policy = net_name(eval_env, args)
		policy.load_network_params(net_params)
		
		avg_reward = 0.
		avg_ep_len = 0
		for _ in range(eval_episodes):
			state = eval_env.reset()
			info = eval_env.get_info(state)
			done = False
			while not done:
				action = policy.select_action(state,info,goal)
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				avg_ep_len += 1
				
		avg_reward /= eval_episodes
		avg_ep_len /= eval_episodes 
		
		print("---------------------------------------")
		print(f"Evaluation over {eval_episodes} episodes:\n")
		print(f"Mean reward: {avg_reward[0]:.3f}") 
		print(f"Mean episode length: {avg_ep_len:.0f}")
		print("---------------------------------------")

		eval_env.close()
		queue.put([avg_reward,avg_ep_len])


class Agent:
	def __init__(self, env, model, writer, args):
		
		# Hyperparameters
		self.args = args

		# GYM Environment
		self.env = env

		# Determine number of goals
		self.num_goals = env.num_goals

		# Selected model
		self.model_name = model
		self.model = model(env, args, writer)

		self.state_dim, self.action_dim, self.max_action = self.model._get_space_parameters(env)

		# Writer
		self.writer = writer

		self.eval_iter = 0
		self.eval_queue = Queue()

	def eval_policy(self, eval_queue, eval_episodes = 10):
		net_params = self.model.get_network_params()
		p = Process(target=run_eval_policy, args=(net_params, self.model_name, self.args, self.args['env'], self.args['seed'], eval_queue, eval_episodes))
		p.start()
		out = eval_queue.get()
		p.join()
		return out


	def learn(self, num_timesteps):

		# Load policy if required
		if self.args['load_model'] != "":
			file_name = f"{'TD3'}_{self.args['env']}_{self.args['seed']}"
			policy_file = file_name if self.args['load_model'] == "default" else self.args['load_model']
			self.load(f"./models/{policy_file}")

		
		# Evaluate untrained policy
		avg_rew,avg_ep_len = self.eval_policy(self.eval_queue, eval_episodes = 1)
		self.writer.add_scalar('Eval/avg_rew', avg_rew, self.eval_iter)
		self.writer.add_scalar('Eval/avg_ep_len', avg_ep_len, self.eval_iter)
		self.eval_iter +=1
		
		# Reset the environment
		state,done = self.env.reset(), False
		info = self.env.get_info(state)

		# Structure to store transition data
		observations, achieved_goals, desired_goals, actions, infos = self._create_transition_buffers() 

		# Initial data register
		observations[0] = state['observation']
		achieved_goals[0] = state['achieved_goal']
		if self.num_goals > 1:
			for i in range(self.num_goals):
				desired_goals[0][i] = state['desired_goal'+str(i+1)]
		else:
			desired_goals[0][0] = state['desired_goal']

		for key, val in info.items():
			infos[key][0] = val

		# Pick last goal for training
		goal = self.num_goals if self.num_goals > 1 else None

		# Initial per episode measurements
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		# Start the training loop
		for t in range(int(num_timesteps)):

			episode_timesteps += 1
			
			# Select action randomly or according to policy
			if t < self.args['start_timesteps']:
				action = self.env.action_space.sample()
			else:
				action = (
					self.model.select_action(state,info,goal)
					+ np.random.normal(0, self.max_action 
						* self.args['expl_noise'], size=self.action_dim)
				).clip(-self.max_action, self.max_action)
			
			# Register the action
			actions[episode_timesteps - 1] = action
				
			# Perform action
			next_state, reward, done, info = self.env.step(action)

			# Determine the done flag
			done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0

			# Register the observations and goals
			observations[episode_timesteps] = next_state['observation']
			achieved_goals[episode_timesteps] = next_state['achieved_goal']
			if self.num_goals > 1:
				for i in range(self.num_goals):
					desired_goals[episode_timesteps][i] = next_state['desired_goal'+str(i+1)]
			else:
				desired_goals[episode_timesteps][0] = next_state['desired_goal']

			for key, val in info.items():
				if key != 'TimeLimit.truncated':
					infos[key][episode_timesteps] = val
				
			# Re-assign values for next iteration
			state = next_state

			# Update episode reward
			episode_reward += reward
			
			# Train agent after collecting sufficient data
			if t >= self.args['start_timesteps']:
				self.model.train(self.args['batch_size'])
			# If episode completes store trajectories and reset
			if done:
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward[0]:.3f}")
				# Log to tensorboard
				self.writer.add_scalar('Train/episode_rew', episode_reward[0], episode_num)
				self.writer.add_scalar('Train/episode_len', episode_timesteps, episode_num)
				# Store trajectories in the buffer
				episode = [np.expand_dims(observations,0), 
					np.expand_dims(achieved_goals,0), 
					np.expand_dims(desired_goals,0), 
					np.expand_dims(actions,0),
					infos]
				self.model.replay_buffer._store_transitions(episode)
				# Reset environment
				state,done = self.env.reset(),False
				info = self.env.get_info()
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1
				# Initial data register
				observations[0] = state['observation']
				achieved_goals[0] = state['achieved_goal']
				if self.num_goals > 1:
					for i in range(self.num_goals):
						desired_goals[0][i] = state['desired_goal'+str(i+1)]
				else:
					desired_goals[0][0] = state['desired_goal']
				
			# Evaluate episode
			if (t + 1) % self.args['eval_freq'] == 0:
				avg_rew,avg_ep_len = self.eval_policy(self.eval_queue)
				self.writer.add_scalar('Eval/avg_rew', avg_rew, self.eval_iter)
				self.writer.add_scalar('Eval/avg_ep_len', avg_ep_len, self.eval_iter)
				self.eval_iter +=1
				if self.args['save_model']: 
					self.model.save(f"./models/"+self.args['env']+'_'+str(t+1))

	def _create_transition_buffers(self):
		observations = np.zeros((self.env._max_episode_steps + 1, self.env.observation_space.spaces['observation'].shape[0]))
		achieved_goals = np.zeros((self.env._max_episode_steps + 1, self.env.observation_space.spaces['achieved_goal'].shape[0]))
		d_goal = [key for key in self.env.observation_space.spaces.keys() if 'desired_goal' in key]
		desired_goals = np.zeros((self.env._max_episode_steps + 1, self.num_goals, self.env.observation_space.spaces[d_goal[0]].shape[0]))
		actions = np.zeros((self.env._max_episode_steps, self.action_dim))
		infos = dict()
		for key, val in self.env.env.info_space.items():
			infos[key] = np.zeros((self.env._max_episode_steps + 1, val.shape[0]))
		return observations, achieved_goals, desired_goals, actions, infos

	def __del__(self):
		self.eval_queue.close()
		self.env.close()


