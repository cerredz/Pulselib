
import torch
import torch.nn as nn
from pathlib import Path
import copy
import torch.optim as optim
import gymnasium as gym
from typing import Optional
from utils.ReplayBuffer import ReplayBufferTorch
from utils.agents import OrnsteinUhlenbeckNoise, load_optimizer, load_weights_path
from enum import Enum

# deep deterministic policy gradient architecture file
class DDPGNoise(Enum):
    Ornstein = 1
    Gaussian = 2

class DDPG(nn.Module):
    def __init__(
        self,
        env_action_space: gym.spaces.Space,
        state_dim: int, 
        device: torch.device,
        gamma: float, 
        criterion: nn.Module,
        learning_rate: float,
        weight_decay: float,
        actor_network: nn.Module,
        critic_network: nn.Module,
        target_actor_network: nn.Module,
        target_critic_network: nn.Module,
        batch_size: int,
        update: int | float, 
        replay_buffer: 'ReplayBufferTorch',
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        target_network: Optional[nn.Module] = None,
        actor_weights_path: Optional[str | Path] = None,
        critic_weights_path: Optional[str | Path] = None,
        noise: Optional[DDPGNoise] = DDPGNoise.Ornstein, 
        reset_weights: bool = False,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = .2,
        tau: float = .001
        ):

        super().__init__()
        assert isinstance(actor_network, nn.Module) and isinstance(critic_network, nn.Module), "Both the actor and critic network must be of type nn.Module for the DDPG agent"
        assert isinstance(criterion, nn.Module), "criterion for DDPG agent must be of instance nn.Module"
        assert isinstance(env_action_space, gym.spaces.Box), "DDPG requires a Box action space (use Box(n) in your env)"
        assert state_dim > 0, "state dimensions cannot be negative"

        # config vars
        self.env_action_space=env_action_space
        self.action_dim = self.env_action_space.shape[0]
        self.device = device
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.actor_network = actor_network.to(device)
        self.critic_network=critic_network.to(device)
        self.update = update
        self.actor_weights_path = actor_weights_path 
        self.gamma=gamma
        self.state_dim=state_dim
        self.criterion=criterion
        self.replay_buffer = replay_buffer
        self.tau = tau
        self.batch_size = batch_size
        self.step = 0 

        self.action_low = torch.tensor(env_action_space.low, device=device, dtype=torch.float32)
        self.action_high = torch.tensor(env_action_space.high, device=device, dtype=torch.float32)

        # actor critic networks
        actor_weights=load_weights_path(actor_weights_path, device)
        if actor_weights: self.actor_network.load_state_dict(actor_weights)
        critic_weights = load_weights_path(critic_weights_path, device)
        if critic_weights: self.critic_network.load_state_dict(critic_weights)

        # zero weights for DPPG
        if reset_weights:
            self.__init_weights() # initialize weightsd to DDPG paper

        self.target_actor_network=copy.deepcopy(self.actor_network).to(self.device) if not target_actor_network else target_actor_network
        self.target_critic_network=copy.deepcopy(self.critic_network).to(self.device) if not target_critic_network else target_critic_network

        # optimizers + noise      
        self.actor_optimizer=load_optimizer(actor_optimizer, self.actor_network.parameters(), self.learning_rate, self.weight_decay)
        self.critic_optimizer=load_optimizer(critic_optimizer, self.critic_network.parameters(), self.learning_rate, self.weight_decay)
        self.noise=OrnsteinUhlenbeckNoise(size=(self.batch_size, self.action_dim, ), mu=mu, theta=theta, sigma=sigma, device=device) # imeplement more noise options later

    def __init_weights(self):
        """Initialize weights according to DDPG paper"""
        for p in list(self.actor_network.parameters())[-2:]:
            nn.init.uniform_(p, -3e-3, 3e-3)
        for p in list(self.critic_network.parameters())[-2:]:
            nn.init.uniform_(p, -3e-4, 3e-4)

    def reset_noise(self):
        self.noise.reset()

    def forward(self, states):
        return self.actor_network(states)

    def action(self, states):
        self.actor_network.eval()
        actions = self.actor_network(states)
        self.actor_network.train()
        actions = actions + self.noise.sample()
        actions = torch.clamp(actions, min=self.action_low, max=self.action_high)
        return actions

    def train_step(self, states, actions, rewards, next_states, dones):

        states = states.detach()
        actions = actions.detach()
        next_states = next_states.detach()
        rewards = rewards.detach()
        dones = dones.detach()

        current_q = self.critic_network(torch.cat([states, actions], dim=1))

        rewards = rewards.view(-1, 1)  # Force [256] -> [256, 1]
        dones = dones.view(-1, 1)      # Force [256] -> [256, 1]

        with torch.no_grad():
            target_actor = self.target_actor_network(next_states)
            target_q_critic = self.target_critic_network(torch.cat([next_states, target_actor], dim=1))
            target = rewards + (1 - dones.float()) * self.gamma * target_q_critic   
            

        # critic loss
        critic_loss = self.criterion(current_q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor loss
        actor_actions = self.actor_network(states)
        actor_loss = -self.critic_network(torch.cat([states, actor_actions], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft updates
        with torch.no_grad():
            for tp, p in zip(self.target_actor_network.parameters(), self.actor_network.parameters()):
                tp.data.lerp_(p.data, self.tau)
            for tp, p in zip(self.target_critic_network.parameters(), self.critic_network.parameters()):
                tp.data.lerp_(p.data, self.tau)

    # save the network and target network somewhere
    #def save(self, network_save_path, target_save_path):
    #    assert Path.exists(network_save_path) and Path.exists(target_save_path)
    #
    #    torch.save(self.network.state_dict(), network_save_path)
    #    torch.save(self.target_network.state_dict(), target_save_path)