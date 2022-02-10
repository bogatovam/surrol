import gym
import surrol
import numpy as np
from matplotlib import pyplot as plt
import torch


from stable_baselines3 import DDPG,PPO,TD3, HerReplayBuffer, VecHerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
import random

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedleReach-v0'
    n_envs = 4
    max_episode_length = 50
    total_timesteps = 6e4
    learning_starts = 10000
    lr = 1e-3
    buffer_size = 200000
    batch_size = 1024
    log_dir = "./logs/TD3/NeedleReach-v0/"
    seed=1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    print('Running for n_procs = {}'.format(n_envs))

    env = make_vec_env(env_id,n_envs,seed,monitor_dir=log_dir,vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=100.)

    #env = gym.make(env_id)
    #env = Monitor(env, log_dir)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3('MultiInputPolicy', 
        env,
        action_noise=action_noise,
        batch_size = batch_size,
        learning_starts = learning_starts,
        buffer_size=buffer_size, 
        replay_buffer_class=VecHerReplayBuffer, 
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=max_episode_length
        ),
        learning_rate=lr,
        train_freq=1,
        gradient_steps=n_envs,
        verbose=0,
        tensorboard_log=log_dir+"./tensorboard/",
        seed=seed)
    

    model.learn(total_timesteps,tb_log_name='run')

    model.save(log_dir+"TD3_HER_NeedleReach-v0")
    env.save(log_dir+"TD3_HER_NeedleReach-v0_stats")








