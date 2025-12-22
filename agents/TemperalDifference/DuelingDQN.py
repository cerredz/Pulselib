from math import nextafter
from re import L
import torch
import torch.nn as nn
from pathlib import Path
import copy
import torch.optim as optim
import gymnasium as gym
from typing import Optional

class DuelingDQN(nn.Module):
    def __init__(
        self,
        env_action_space: gym.spaces.Space,
        state_dim: int, 
        device: torch.device,
        gamma: float, 
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
        self.gamma=gamma
        self.optimizer=optimizer
        self.state_dim=state_dim
        self.criterion=criterion
        hidden_layer_size = self._get_hidden_layer_size()
        self.value_stream = nn.Linear(hidden_layer_size, 1).to(device)
        self.advantage_stream = nn.Linear(hidden_layer_size, self.action_dim).to(device)
        self.step=0

        if optimizer == None:
            # default optimizer, AdamW
            self.optimizer=optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if weights_path is not None:
            assert isinstance(weights_path, (str, Path)), "weights_path must be a string or pathlib.Path"
            path = Path(weights_path)
            assert path.suffix == '.pth', "weights_path must end with '.pth'"
            assert path.exists(), "weights_path does not exist on disk"
            self.network.load_state_dict(torch.load(weights_path, map_location=device))
            if target_network is None:
                self.target_network=copy.deepcopy(self.network)

        self.target_network = copy.deepcopy(self.network).to(self.device) if not target_network else target_network.to(device)

    # utility function to read the passed in network and build the adantage function layer for the 
    # DoubleDQN network (need to get the size of the last hidden layer in the passed in network)
    def _get_hidden_layer_size(self):
        """Utility function to get the output size of the last layer in the network"""
        linear_layers = [module for module in self.network.modules() if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise ValueError("Network must contain at least one Linear layer")
        return linear_layers[-1].out_features

    def forward(self, states):
        hidden = self.network(states)
        v = self.value_stream(hidden)
        a = self.advantage_function(hidden)
        return v + (a - a.mean(dim=1, keepdim=True))

    def decay_epsilon(self):
        self.epsilon=max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def action(self, states):
        states=states.float().to(self.device)
        probs=torch.rand(states.shape[0], dtype=torch.float32, device=self.device)
        explore_mask=(probs < self.epsilon)
        e_greedy_actions=torch.randint(0, self.action_dim, (states.shape[0],), device=self.device, dtype=torch.int64)
        
        self.eval()
        with torch.inference_mode():
            q_values=self(states)
            q_value_actions=torch.argmax(q_values, dim=-1)

        actions=torch.where(explore_mask, e_greedy_actions, q_value_actions)
        self.train()
        return actions

    def train_step(self, states, actions, rewards, next_states, dones):
        all_q_values=self.forward(states)
        pred_q = all_q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
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

    def __repr__(self):
        pass