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
        env = gym.make(env_id,render_mode='human')
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedleReach-v0'
    max_episode_length = 50
    total_timesteps = 4e4
    learning_starts = 20000
    lr = 1e-3
    buffer_size = 200000
    batch_size = 2048
    log_dir = "./logs/TD3/NeedleReach-v0/"
    seed=1

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    env = make_vec_env(env_id,1,seed,monitor_dir=log_dir,env_kwargs={'render_mode':'human','seed':seed})

    env = VecNormalize(env,norm_obs=True)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3('MultiInputPolicy', 
        env,
        action_noise=action_noise,
        batch_size = batch_size,
        learning_starts = learning_starts,
        buffer_size=buffer_size, 
        replay_buffer_class=HerReplayBuffer, 
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=max_episode_length,
            handle_timeout_termination=True
        ),
        learning_rate=lr,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir+"./tensorboard/",
        seed=seed)
    

    model.learn(total_timesteps,tb_log_name='run')

    model.save(log_dir+"TD3_HER_NeedleReach-v0")
    env.save(log_dir+"TD3_HER_NeedleReach-v0_stats")








