import numpy as np

class her_sampler:
    def __init__(self, replay_k, num_goals, reward_func=None, done_func = None):

        # Number of augmented samples per batch
        self.replay_k = replay_k

        # Proportion of her samples
        self.future_p = 1 - (1. / (1 + replay_k))

        # Number of goals
        self.num_goals = num_goals

        # Final goal to use
        if self.num_goals > 1:
            self.final_goal = 'desired_goal'+str(self.num_goals)
        else:
            self.final_goal = 'desired_goal'

        # Reward func
        self.reward_func = reward_func

        # Done func
        self.done_func = done_func

    def sample_her_transitions(self, buffer, batch_size):

        # Trajectory lenght
        trajectory_length = buffer['action'].shape[1]

        # Buffer size
        buffer_lenght = buffer['action'].shape[0]

        # generate idxs which trajectories to use
        episode_idxs = np.random.randint(low = 0, high = buffer_lenght, size = batch_size)

        # generate idxs which timesteps to use
        t_samples = np.random.randint(low = 0, high = trajectory_length, size = batch_size)

        # Copy data from the buffer according to determined indexes
        batch = {'info': dict(), 'next_info': dict()}
        for key in buffer.keys():
            if key == 'info' or key == 'next_info':
                for info_key in buffer[key].keys():
                    batch[key][info_key] = buffer[key][info_key][episode_idxs, t_samples].copy()
            else:
                batch[key] = buffer[key][episode_idxs, t_samples].copy()


        # Determine which transitions to use for HER augmentation
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)

        # Sample 'future' timesteps for each 't_samples'
        future_offset = np.random.uniform(size=batch_size) * (trajectory_length - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Get the achieved_goal at the 'future' timesteps
        next_achieved_goal = buffer['achieved_goal'][episode_idxs[her_indexes], future_t]

        # Replace the 'desired_goal' with the 'next_achieved_goal'
        batch[self.final_goal][her_indexes] = next_achieved_goal
        
        # Recompute the reward for the augmented 'desired_goal'
        batch['reward'] = self.reward_func(batch['next_achieved_goal'], batch[self.final_goal], batch['info'])

        # Recompute the termination state for the augmented 'desired_goal'
        batch['done'] = self.done_func(batch['next_achieved_goal'], batch[self.final_goal], batch['info'])

        # Reshape the batch
        for key in batch.keys():
            if key == 'info' or key == 'next_info':
                for key_info in batch[key].keys():
                    batch[key][key_info] = batch[key][key_info].reshape(batch_size, *batch[key][key_info].shape[1:])
            else:
                batch[key] = batch[key].reshape(batch_size, *batch[key].shape[1:])



        return batch