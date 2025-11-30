import gymnasium as gym
import numpy as np
import numba


class TetrisNumbaEnv(gym.Env):
    def __init__(self):
        self.board=np.zeros((10,10), dtype=np.int32)

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        pass