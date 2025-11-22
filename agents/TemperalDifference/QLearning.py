import gymnasium as gym
import random
from collections import defaultdict
import numpy as np
from numba import njit, float64, int32
from utils.numba import select_action_epsilon_greedy_numba, update_q_entry

class QLearning:
    def __init__(self, env: gym.Env, gamma: float, step_size: float, epsilon: float):
        # Validation
        assert 0 < step_size <= 1, 'step size must be 0 < x <= 1'
        assert epsilon > 0, 'epsilon must be > 0'
        
        self.env = env
        self.n = env.action_space.n
        self.alpha = step_size
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-Table: Maps state tuple -> Array of float64 (one for each action)
        self.q = defaultdict(lambda: np.zeros(self.n, dtype=np.float64))

    def learn(self, num_episodes):
        assert num_episodes > 0
        
        for i in range(num_episodes):   
            r_state, info = self.env.reset()
            state = tuple(r_state.flatten())
            terminated, truncated = False, False

            while not terminated and not truncated:
                action = select_action_epsilon_greedy_numba(self.q[state], self.epsilon)
                
                next_r_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = tuple(next_r_state.flatten())
                
                update_q_entry(
                    self.q[state],
                    action,         
                    self.q[next_state], 
                    self.alpha, 
                    float(reward),
                    self.gamma,
                    terminated
                )
                
                state = next_state