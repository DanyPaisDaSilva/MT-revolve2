from typing import Optional, Union, List

import gym
from gym import spaces
from gym.core import RenderFrame
import numpy as np


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()
        # spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # get action & observation space from Individual
        # TODO: requires pointer to Individual
        self.action_space = None
        self.observation_space = None

        # TODO: save starting state


    def reset(self):
        # reset
        # TODO: return to starting state
        pass

    def step(self, action):
        # take action and calc reward

        # TODO: needs access to current state
        # TODO: ability to take action
        # TODO: access to new state

        # current state
        # take action
        # new state
        # reward = compare position (current state, new state)

        # check if finished

        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        # not needed
        pass

    def close(self):
        # not needed? maybe mujoco needs to close or something
        pass

    def seed(self, seed):
        # not needed? created with MuJoCo env?
        pass