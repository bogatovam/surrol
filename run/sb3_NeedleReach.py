import gym
import surrol
import numpy as np
from matplotlib import pyplot as plt


from stable_baselines3 import DDPG,PPO,TD3, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

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
    #n_envs = 2
    max_episode_length = 50
    total_timesteps = 3e5
    lr = 5e-4
    buffer_size = 500000
    batch_size = 1024

    #print('Running for n_procs = {}'.format(n_envs))
    
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(n_cpu)])

    #env = make_vec_env(env_id,n_envs)

    env = gym.make(env_id)

    env.reset()

    model = TD3('MultiInputPolicy', 
        env,
        batch_size = batch_size,
        learning_starts = 10000,
        buffer_size=buffer_size, 
        replay_buffer_class=HerReplayBuffer, 
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=max_episode_length,
        ),
        learning_rate=lr,
        verbose=1,
        tensorboard_log="./tb/HER_NeedleReach/")
    

    model.learn(total_timesteps,tb_log_name='HER_test')

    model.save("./her_env")
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    

    model = DDPG.load('./her_env', env=env)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()

        if done:
            obs = env.reset()











