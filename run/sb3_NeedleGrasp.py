import os
import gym
import numpy as np
from matplotlib import pyplot as plt
import stable_baselines3
import torch

import surrol
from surrol.algorithms import TD3SharedCrtitic
from surrol.algorithms.buffers import HerMultiGoal

from stable_baselines3 import TD3
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import random

class ModelEnvCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(ModelEnvCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            self.model.env.save(path+'_stats')
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

if __name__ == '__main__':

    
    ############################################
    ############## PARAMETERS ##################
    ############################################

    env_id = 'NeedlePick-v0'
    max_episode_length = 50
    total_timesteps = 7e5
    save_frequency = 50000
    learning_starts = 100
    lr = 4e-5
    buffer_size = 200000
    batch_size = 2048
    log_dir = "./logs/TD3/"+env_id+"/"
    seed=1
    tau = 0.01
    gamma = 0.95

    env = make_vec_env(env_id,1,seed,monitor_dir=log_dir,env_kwargs={'render_mode':'humans'})

    env = VecNormalize(env,norm_obs=False,norm_reward=False)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3SharedCrtitic('MultiInputPolicy', 
        env,
        action_noise=action_noise,
        batch_size = batch_size,
        replay_buffer_class=HerMultiGoal,
        learning_starts = learning_starts,
        policy_kwargs= dict(net_arch=[512, 1024, 512]),
        buffer_size=buffer_size,  
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=max_episode_length,
            handle_timeout_termination=True
        ),
        learning_rate=lr,
        tau=tau,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir+"./tensorboard/",
        seed=seed)
    
    checkpoint_callback = ModelEnvCheckpointCallback(save_freq=save_frequency, save_path=log_dir,
                                         name_prefix='TD3_HER_'+env_id)

    model.learn(total_timesteps,callback=checkpoint_callback,tb_log_name='run')

    model.save(log_dir+"TD3_HER_"+env_id)
    env.save(log_dir+"TD3_HER_"+env_id+"_stats")








