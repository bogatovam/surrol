import gym
import surrol
import numpy as np
from matplotlib import pyplot as plt
import time


from stable_baselines3 import DDPG,PPO,TD3, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedleGrasp-v0'
    model_dir = "./logs/TD3/NeedleGrasp-v0/"
    log_dir = "./logs/TD3/NeedleGrasp-v0/tensorboard/eval"

    seeds = [2,4,9]

    for run,seed in enumerate(seeds):

        writer = SummaryWriter(log_dir+str(run))

        env = make_vec_env(env_id=env_id,n_envs=1,seed=seed,monitor_dir=model_dir,env_kwargs={'render_mode':'human'})

        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False

        model = TD3.load(model_dir+'TD3_HER_'+env_id+'', env=env, kwargs={'seed': seed})

        obs = env.reset()

        for trial in range(100):
            done = [False]
            obs = env.reset()
            steps = 0
            total_reward = 0
            success = False
            while not done[0] and steps < 50:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                steps += 1
                total_reward += reward[0]
                success = info[0]['is_success']

            
                writer.add_scalar('Total reward', total_reward,trial)
                writer.add_scalar('Episode length', steps,trial)
                writer.add_scalar('Success', success, trial)
        
        writer.close()

        env.close()
          

            











