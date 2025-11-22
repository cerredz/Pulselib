import gymnasium as gym
from collections import defaultdict
import numpy as np
from numba import njit, float64, int32

# Import your optimized Numba kernels here
# Assuming they are in the same file or imported from utils
from utils.numba import select_action_epsilon_greedy_numba, update_q_entry

class QLearning:
    def __init__(self, env: gym.Env, config: dict):
        self.env = env
        self.n = env.action_space.n
        
        # Unpack config
        self.alpha = config["ALPHA"]
        self.epsilon = config["EPSILON"]
        self.gamma = config["GAMMA"]
        
        # Q-Table
        self.q = defaultdict(lambda: np.zeros(self.n, dtype=np.float64))

    def get_action(self, state):
        """Returns the action based on current state and epsilon."""
        # Pass the specific Q-values for this state to Numba
        return select_action_epsilon_greedy_numba(self.q[state], self.epsilon)

    def update(self, state, action, next_state, reward, terminated):
        """Updates the Q-table based on the transition."""
        update_q_entry(
            self.q[state],
            action,
            self.q[next_state], 
            self.alpha, 
            float(reward),
            self.gamma,
            terminated
        )