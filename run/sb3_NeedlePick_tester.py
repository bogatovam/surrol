import gym
import surrol
import numpy as np
from matplotlib import pyplot as plt
import time


from stable_baselines3 import DDPG,PPO,TD3, HerReplayBuffer, VecHerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedlePickPointSpecific-v0'
    log_dir = "./logs/TD3/"+env_id+'/'
    seed = 1

    env = make_vec_env(env_id,1,seed,monitor_dir=log_dir,env_kwargs={'render_mode':'human','seed':seed})
    env = VecNormalize.load(log_dir+"TD3_HER_"+env_id+"_stats", env)

    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    model = TD3.load(log_dir+'TD3_HER_'+env_id+'', env=env)

    for _ in range(30):

        obs = env.reset()
        done = [False]
        while not done[0] or not bool(info[0]["is_success"]):
            time.sleep(0.1)
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
          

            











