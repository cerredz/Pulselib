from math import nextafter
from re import L
import torch
import torch.nn as nn
from pathlib import Path
import copy
import torch.optim as optim
import gymnasium as gym
from typing import Optional
from utils.ReplayBuffer import ReplayBufferTorch
from utils.agents import load_optimizer, load_weights_path

# deep deterministic policy gradient architecture file

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
        epsilon: float,
        update: int | float, 
        epsilon_decay: float,
        epsilon_min: float,
        replay_buffer: 'ReplayBufferTorch',
        optimizer: Optional[torch.optim.Optimizer] = None,
        target_network: Optional[nn.Module] = None,
        actor_weights_path: Optional[str | Path] = None,
        critic_weights_path: Optional[str | Path] = None
    ):
        super().__init__()
        assert isinstance(actor_network, nn.Module) and isinstance(critic_network, nn.Module), "Both the actor and critic network must be of type nn.Module for the DDPG agent"
        assert isinstance(criterion, nn.Module), "criterion for DQN agent must be of instance nn.Module"
        assert isinstance(env_action_space, gym.spaces.Discrete), "DQN requires a Discrete action space (use Discrete(n) in your env)"
        assert state_dim > 0, "state dimensions cannot be negative"
        assert isinstance(replay_buffer, 'ReplayBufferTorch'), 'replay buffer must be of type PulseLib replay buffer torch'

        self.env_action_space=env_action_space
        self.action_dim=int(self.env_action_space.n)
        self.device = device
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.actor_network = actor_network.to(device)
        self.critic_network=critic_network.to(device)
        self.epsilon = epsilon
        self.update = update
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.actor_weights_path = actor_weights_path 
        self.optimizer=optimizer
        self.gamma=gamma
        self.state_dim=state_dim
        self.criterion=criterion
        self.replay_buffer = replay_buffer
        self.optimizer=load_optimizer(optimizer, self.actor_network.parameters(), learning_rate, weight_decay)

        actor_weights=load_weights_path(actor_weights_path, device)
        if actor_weights: self.actor_network.load_state_dict(actor_weights)
                
        critic_weights = load_weights_path(critic_weights_path, device)
        if critic_weights: self.critic_network.load_state_dict(critic_weights)

        self.target_actor_network=copy.deepcopy(self.actor_network).to(self.device) if not target_actor_network else target_actor_network
        self.target_critic_network=copy.deepcopy(self.critic_network).to(self.device) if not target_critic_network else target_critic_network

        self.step=0


    def forward(self, states):
        return self.network(states)

    def decay_epsilon(self):
        self.epsilon=max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def action(self, states):
        states=states.float().to(self.device)
        probs=torch.rand(states.shape[0], dtype=torch.float32, device=self.device)
        explore_mask=(probs < self.epsilon)
        e_greedy_actions=torch.randint(0, self.action_dim, (states.shape[0],), device=self.device, dtype=torch.int64)
        
        self.network.eval()
        with torch.inference_mode():
            q_values=self.network(states)
            q_value_actions=torch.argmax(q_values, dim=-1)

        actions=torch.where(explore_mask, e_greedy_actions, q_value_actions)
        self.network.train()
        return actions

    def train_step(self, states, actions, rewards, next_states, dones):
        states=states.float()
        next_states=next_states.float()

        all_q_values=self.forward(states)
        pred_q = all_q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size,)
        
        with torch.no_grad():
            q_values_next=self.target_network(next_states)
            max_q_next=q_values_next.max(dim=1).values
            true_q=rewards + (1 - dones.float()) * self.gamma * max_q_next
        
        loss=self.criterion(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % 1000 == 0:
            print(f"Step {self.step} | Loss: {loss.item():.4f} | Epsilon: {self.epsilon:.4f}")

        if self.step % self.update == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    # save the network and target network somewhere
    def save(self, network_save_path, target_save_path):
        assert Path.exists(network_save_path) and Path.exists(target_save_path)

        torch.save(self.network.state_dict(), network_save_path)
        torch.save(self.target_network.state_dict(), target_save_path)