import gym
import surrol
import numpy as np
from matplotlib import pyplot as plt
import time


from stable_baselines3 import DDPG,PPO,TD3, HerReplayBuffer, VecHerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from surrol.tasks import NeedleReach


if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedleReach-v0'
    log_dir = "./logs/TD3/NeedleReach-v0/"

    
    #env = NeedleReach(render_mode='human')
    env = SubprocVecEnv([lambda: NeedleReach(render_mode='human')])
    env = VecNormalize.load(log_dir+"TD3_HER_NeedleReach-v0_stats", env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    model = TD3.load(log_dir+'TD3_HER_NeedleReach-v0', env=env)

    obs = env.reset()

    for _ in range(100):
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(0.1)

        if done[0] or bool(info[0]["is_success"]):
            obs = env.reset()











