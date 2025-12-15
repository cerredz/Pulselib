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

        if optimizer:
            assert isinstance(optimizer, optim.Optimizer), "optimizer must be a torch.optim.Optimizer instance"

        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.network = network.to(device)
        self.epsilon = epsilon
        self.update = update
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.weights_path = weights_path 
        self.optimizer=optimizer
        self.q_learning_rate
        self.optimizer=optimizer

        if optimizer == None:
            self.optimizer=optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if target_network is not None:
            self.target_network = target_network.to(device)

        if weights_path is not None:
            assert isinstance(weights_path, (str, Path)), "weights_path must be a string or pathlib.Path"
            path = Path(weights_path)
            assert path.suffix == '.pth', "weights_path must end with '.pth'"
            assert path.exists(), "weights_path does not exist on disk"
            self.network.load_state_dict(torch.load(weights_path, map_location=device))
            if target_network is None:
                self.target_network=copy.deepcopy(self.network)

    def forward(self, states):
        return self.network(states)

    def action(self, states):
        self.epsilon=max(self.epsilon_min, self.epsilon * self.epsilon_decay)




    def train_step(self, states, actions, rewards, next_states, dones):
        q_values=self.forward(states)
        print(q_values)
        print(q_values.shape)
