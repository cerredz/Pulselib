from math import nextafter
from re import L
import torch
import torch.nn as nn
from pathlib import Path
import copy
import torch.optim as optim
import gymnasium as gym
from typing import Optional

# an implementation of a reusable deep neural network that predicts q-values for reinforcement learning environemnts
# goal is to make it abstract enough to basically just plug into different training scripts

# can load in a weights file, can pass in the network, target network, optimizer, and loss function

class DQN(nn.Module):
    def __init__(
        self,
        env_action_space: gym.spaces.Space,
        state_dim: int, 
        device: torch.device,
        q_learning_rate: float, 
        criterion: nn.Module,
        learning_rate: float,
        weight_decay: float,
        network: nn.Module,
        epsilon: float,
        update: int | float, 
        epsilon_decay: float,
        epsilon_min: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        target_network: Optional[nn.Module] = None,
        weights_path: Optional[str | Path] = None,
    ):
        super().__init__()
        assert isinstance(network, nn.Module), "Both the network and target network must be of type nn.Module for the DQN agent"
        assert isinstance(criterion, nn.Module), "criterion for DQN agent must be of instance nn.Module"
        assert isinstance(env_action_space, gym.spaces.Discrete), "DQN requires a Discrete action space (use Discrete(n) in your env)"
        assert state_dim > 0, "state dimensions cannot be negative"

        if optimizer:
            assert isinstance(optimizer, optim.Optimizer), "optimizer must be a torch.optim.Optimizer instance"

        self.env_action_space=env_action_space
        self.action_dim=int(self.env_action_space.n)
        self.device = device
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.network = network.to(device)
        self.epsilon = epsilon
        self.update = update
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.weights_path = weights_path 
        self.optimizer=optimizer
        self.q_learning_rate=q_learning_rate
        self.optimizer=optimizer
        self.state_dim=state_dim
        self.criterion=criterion

        if optimizer == None:
            self.optimizer=optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if weights_path is not None:
            assert isinstance(weights_path, (str, Path)), "weights_path must be a string or pathlib.Path"
            path = Path(weights_path)
            assert path.suffix == '.pth', "weights_path must end with '.pth'"
            assert path.exists(), "weights_path does not exist on disk"
            self.network.load_state_dict(torch.load(weights_path, map_location=device))
            if target_network is None:
                self.target_network=copy.deepcopy(self.network)

        if target_network is not None:
            self.target_network = target_network.to(device)
        else:
            self.target_network=copy.deepcopy(self.network).to(self.device)

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
            true_q=rewards + (1 - dones.float()) * self.q_learning_rate * max_q_next
        
        loss=self.criterion(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % 1000 == 0:
            print(f"Step {self.step} | Loss: {loss.item():.4f} | Epsilon: {self.epsilon:.4f}")

        if self.step % self.update == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        

