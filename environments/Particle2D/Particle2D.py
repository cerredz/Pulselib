import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class Particle2D(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, device, batch_size, dt=0.1, max_steps=200):
        super().__init__()
        self.device, self.batch_size, self.dt, self.max_steps = device, batch_size, dt, max_steps
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)
        
    def reset(self, seed=None, options=None):
        self.state = torch.cat([torch.randn(self.batch_size, 2, device=self.device) * 5, 
                                torch.zeros(self.batch_size, 2, device=self.device)], dim=1)
        self.terminated = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.steps = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        return self.state.clone(), {}
    
    def step(self, action):
        action = torch.clamp(action, -1, 1)
        self.state[:, 2:4] += action * self.dt  # velocity += acceleration * dt
        self.state[:, :2] += self.state[:, 2:4] * self.dt  # position += velocity * dt
        dist = torch.norm(self.state[:, :2], dim=1)
        rewards = -dist - 0.001 * torch.sum(action ** 2, dim=1)
        self.steps += 1
        self.terminated = (dist < 0.1) | (self.steps >= self.max_steps)
        return self.state.clone(), rewards, self.terminated, torch.zeros_like(self.terminated), {}
