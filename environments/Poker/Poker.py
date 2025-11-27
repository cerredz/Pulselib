import gynasium as gym
import numpy as np
import math

"""
- Poker environment implementation for rl:
"""

class Poker(gym.Env):
    def __init__(self, n: int, sm: int, bb: int):
        assert n > 1 and n < 10, "Number of players must be between 2 and 9"
        assert sm > 0 and bb > 0, "Small blind and big blind must be greater than 0"
        
        self.action_space=gym.spaces.Discrete(4)

    def _get_obs(self):
        pass
    
    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass