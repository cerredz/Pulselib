import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Optional, Any
import torch

def default_ddpg_actor_critic(env, state_dim):
    n_actions = env.action_space.n

    actor_network=nn.Sequential(
        nn.Linear(state_dim, 32), nn.ReLU(), 
        nn.Linear(32, n_actions), nn.ReLU(), nn.Linear(n_actions, 1)
    )

    critic_network=nn.Sequential(
        nn.Linear(state_dim, 32), nn.ReLU(), 
        nn.Linear(32, 1)
    )

    return actor_network, critic_network

def default_actor_critic_params(env, state_dim, device=torch.device("cpu")):
    n_actions = env.action_space.n
    
    actor_network=nn.Sequential(
        nn.Linear(state_dim, 32), nn.ReLU(), 
        nn.Linear(32, n_actions), nn.Softmax(dim=-1)
    )

    critic_network=nn.Sequential(
        nn.Linear(state_dim, 32), nn.ReLU(), 
        nn.Linear(32, 1)
    )

    return {
        'env_action_space': env.action_space,
        'state_dim': state_dim,
        'device': device,
        'actor_network': actor_network,
        'critic_network': critic_network,
        'gamma': 0.99,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4
    }

# basic utility function to either validate or assign
# default optimizer to agents (AdamW)
def load_optimizer(optimizer, parameters, learning_rate, weight_decay):
    if optimizer is None:
        return optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)

    if isinstance(optimizer, optim.Optimizer):
            return optimizer    

    if isinstance(optimizer, type) and issubclass(optimizer, optim.Optimizer):
        return optimizer(parameters, lr=learning_rate, weight_decay=weight_decay)

    raise TypeError(
        "optimizer must be None, a torch.optim.Optimizer instance, "
        "or a torch.optim.Optimizer class"
    )

# utility function to load in a weights path
# input could be None
def load_weights_path(weights_path, device) -> Optional[Dict[str, Any]] :
    if weights_path is not None:
        assert isinstance(weights_path, (str, Path)), "weights_path must be a string or pathlib.Path"
        path = Path(weights_path)
        assert path.suffix == '.pth', "weights_path must end with '.pth'"
        assert path.exists(), f"weights_path does not exist on disk: {path}"
        return torch.load(path, map_location=device, weights_only=True)
    
    return None

class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, device='cpu'):
        self.mu = mu * torch.ones(size, device=device)
        self.theta = theta
        self.sigma = sigma
        self.state = torch.zeros(size, device=device)
        self.device = device
    
    def sample(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * torch.randn_like(self.state)
        return self.state
    
    def reset(self):
        self.state.zero_()