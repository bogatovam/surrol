import gym
from gym import error
from surrol.gym.surrol_env import SurRoLEnv


class SurRoLGoalEnv(SurRoLEnv):
    """
    A gym GoalEnv wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')

        if self.multi_goal:
            keys = ['observation', 'achieved_goal', 'desired_goal1', 'desired_goal2']
        else:
            keys = ['observation', 'achieved_goal', 'desired_goal']
        for key in keys:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
        return super().reset()
