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
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedleGrasp-v0'
    model_dir = "./logs/TD3/NeedleGrasp-v0/"
    log_dir = "./logs/TD3/NeedleGrasp-v0/tensorboard/eval"

    seeds = [2,3,9]

    for run,seed in enumerate(seeds):

        writer = SummaryWriter(log_dir+str(run))

        env = make_vec_env(env_id=env_id,n_envs=1,seed=seed,monitor_dir=model_dir,env_kwargs={'render_mode':'human'})

        #  do not update them at test time
        env.training = False
        
        # reward normalization is not needed at test time
        env.norm_reward = False

        model = TD3.load(model_dir+'TD3_HER_'+env_id+'', env=env, kwargs={'seed': seed})

        rewards,ep_lens = evaluate_policy(model=model,
                            env=env,
                            n_eval_episodes=50,
                            deterministic=True,
                            render=True,
                            return_episode_rewards=True)


        for trial,(r,l) in enumerate(zip(rewards,ep_lens)):

            writer.add_scalar('Total reward', r, trial)
            writer.add_scalar('Episode length', l, trial)

        
        writer.close()

        env.close()
          

            











