import os
import numpy as np
import torch
import gym
import yaml
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from surrol.algorithms import Agent
from surrol.algorithms.TD3 import TD3, TD3MultiGoal


if __name__ == "__main__":

	# Config file
	cfg_file = 'NeedlePickViaGrasp-v0.yaml'

	# Required to start pybullet on a seperate thread
	mp.set_start_method('spawn')

	# Load the hyperparameters
	with open(os.path.dirname(os.path.realpath(__file__)) + '/cfg/' + cfg_file) as f:
		args = yaml.safe_load(f)
	
	# Print initial setting
	file_name = f"{'TD3'}_{args['env']}_{args['seed']}"
	print("---------------------------------------")
	print(f"Policy: {'TD3'}, Env: {args['env']}, Seed: {args['seed']}")
	print("---------------------------------------")
	
	# Create required dirs
	if not os.path.exists("./logs"):
		os.makedirs("./logs")
	if args['save_model'] and not os.path.exists("./models"):
		os.makedirs("./models")
	
	# Create the environment
	env_kwargs = {"render_mode": args['render_mode']}
	env = gym.make(args['env'], **env_kwargs)
	
	# Set seeds
	env.seed(args['seed'])
	env.action_space.seed(args['seed'])
	torch.manual_seed(args['seed'])
	np.random.seed(args['seed'])

	# Create tensorboard instance
	writer = SummaryWriter(log_dir='logs/'+args['env'] + '/run2')

	# Create policy
	if args['model'] == 'TD3':
		policy = Agent(env, TD3, writer, args)
	elif args['model'] == 'TD3MultiGoal':
		policy = Agent(env, TD3MultiGoal, writer, args)
	else:
		raise ValueError

	policy.learn(args['max_timesteps'])
	

	
   

	

	
	

	
		