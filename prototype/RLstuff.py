from typing import Optional, Union, List

import gym
from gym import spaces
from gym.core import RenderFrame
import numpy as np


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()
        # spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = None
        self.observation_space = None


    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass

    def close(self):
        pass

    def seed(self, seed):
        pass