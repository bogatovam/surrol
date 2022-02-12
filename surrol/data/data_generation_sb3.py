"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH
from stable_baselines3 import TD3, HerReplayBuffer, VecHerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, default='NeedleReach-v0',
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

actions = []
observations = []
infos = []

images = []  # record video
masks = []


def main():

    env = gym.make(args.env, render_mode=None)  # 'human'

    model = TD3('MultiInputPolicy', 
        env,
        batch_size = 64,
        buffer_size=200000, 
        replay_buffer_class=VecHerReplayBuffer, 
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=50
        )
    )

    model.policy.set_training_mode(False)

    num_itr = 2 if not args.video else 1
    cnt = 0
    init_state_space = 'random'
    env.reset()
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = 50

    print()
    while cnt < num_itr:
        obs = env.reset()
        print("ITERATION NUMBER ", cnt)
        goToGoal(env,model,obs)
        cnt += 1

    file_name = "data_"
    file_name += args.env
    file_name += "_" + init_state_space
    file_name += "_" + str(num_itr)
    file_name += ".npz"

    folder = 'demo' if not args.video else 'video'
    folder = os.path.join(ROOT_DIR_PATH, 'data', folder)

    #np.savez_compressed(os.path.join(folder, file_name),
                        #acs=actions, obs=observations, info=infos)  # save the file

    print("Replay buffer length: {}".format(model.replay_buffer.size()))
    model.save_replay_buffer(os.path.join(folder, file_name))

    if args.video:
        video_name = "video_"
        video_name += args.env + ".mp4"
        writer = imageio.get_writer(os.path.join(folder, video_name), fps=20)
        for img in images:
            writer.append_data(img)
        writer.close()

        if len(masks) > 0:
            mask_name = "mask_"
            mask_name += args.env + ".npz"
            np.savez_compressed(os.path.join(folder, mask_name),
                                masks=masks)  # save the file

    used_time = time.time() - init_time
    print("Saved data at:", folder)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    print(f"Trials: {num_itr}/{cnt}")
    env.close()


def goToGoal(env,model,last_obs):
    episode_acs = []
    episode_obs = []
    episode_info = []
    episode_done = []
    episode_rew = []

    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
    #episode_obs.append(last_obs)

    obs, success = last_obs, False

    while time_step < min(50, args.steps):
        action = env.get_oracle_action(obs)
        if args.video:
            # img, mask = env.render('img_array')
            img = env.render('rgb_array')
            images.append(img)
            # masks.append(mask)

        obs, reward, done, info = env.step(action)
        # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
        time_step += 1

        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True

        episode_acs.append(action)
        episode_done.append(done)
        episode_obs.append(obs)
        episode_info.append(info)
        episode_rew.append(reward)

    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))

    if success:

        for action,obs,reward,done,info in zip(episode_acs,episode_obs,episode_rew,episode_done,episode_info):
            # Retrieve reward and episode length if using Monitor wrapper
            #model._update_info_buffer(info, done)

            # Store data in replay buffer (normalized action and unnormalized observation)
 
            if 'TimeLimit.truncated' in info.keys():
                info.pop('TimeLimit.truncated')
            if done:
                print()
            model._update_info_buffer([info], [done])
            #model._store_transition(model.replay_buffer, action, obs, reward, done, info)


        #actions.append(episode_acs)
        #observations.append(episode_obs)
        #infos.append(episode_info)


if __name__ == "__main__":
    main()
