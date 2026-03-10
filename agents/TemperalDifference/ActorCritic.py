import torch
import torch.nn as nn
from pathlib import Path
import copy
import torch.optim as optim
import gymnasium as gym
from typing import Optional

## class that implements actor-critic algorithm in rl

# can pass in your model architectures

class ActorCritic(nn.Module):
    def __init__(
        self,
        env_action_space: gym.spaces.Space,
        state_dim: int, 
        device: torch.device,
        actor_network: nn.Module,
        critic_network: nn.Module,
        gamma: float = 0.99, 
        critic_criterion: nn.Module = nn.MSELoss(), # Default to MSE for the Critic
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        weights_path: Optional[str | Path] = None,
        train_log_freq: int=1000
    ):
        super().__init__()
        
        assert isinstance(actor_network, nn.Module) and isinstance(critic_network, nn.Module), \
            "Actor and Critic networks must be of type nn.Module"
        assert isinstance(critic_criterion, nn.Module), "Critic criterion must be an instance of nn.Module"
        assert isinstance(env_action_space, gym.spaces.Discrete), \
            "This ActorCritic implementation assumes a Discrete action space."
        assert state_dim > 0, "State dimensions must be positive."

        if optimizer:
            assert isinstance(optimizer, optim.Optimizer), "Optimizer must be a torch.optim.Optimizer instance"

        self.env_action_space = env_action_space
        self.action_dim = int(self.env_action_space.n)
        self.device = device
        self.gamma = gamma
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.critic_criterion = critic_criterion
        
        self.actor = actor_network.to(self.device)
        self.critic = critic_network.to(self.device)

        self.optimizer = optimizer
        if self.optimizer is None:
            # Combine parameters from both Actor and Critic into one list
            all_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
            self.optimizer = optim.AdamW(
                all_parameters, 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )

        if weights_path is not None:
            self._load_weights(weights_path)

        self.step = 0
        self.train_log_freq=train_log_freq

    def _load_weights(self, path_str):
        """Helper to load weights safely."""
        path = Path(path_str)
        if path.exists() and path.suffix == '.pth':
            checkpoint = torch.load(path, map_location=self.device)
            try:
                self.actor.load_state_dict(checkpoint.get('actor', checkpoint))
                self.critic.load_state_dict(checkpoint.get('critic', checkpoint))
                print(f"Loaded weights from {path}")
            except RuntimeError as e:
                print(f"Error loading weights: {e}. Check architecture matches checkpoint.")
        else:
            print(f"Warning: Weights path {path} not found or invalid extension.")

    def forward(self, states):
        action_probs = self.actor(states)
        val = self.critic(states)
        return action_probs, val

    def action(self, states):
        states=states.float()
        self.actor.eval()
        with torch.inference_mode():
            action_probs=self.actor(states)
        self.actor.train()
        action_vals=torch.multinomial(action_probs, 1, replacement=True)
        return action_vals.flatten()

    def train_step(self, states, actions, rewards, next_states, dones):
        pred_action_probs, pred_critic_val = self.forward(states)

        rewards=rewards.view(-1, 1).to(self.device)
        dones = dones.view(-1, 1).to(self.device)

        with torch.no_grad():
            next_critic_states=self.critic(next_states)
            critic_target=rewards + (1-dones.float()) * self.gamma * next_critic_states
            a=(critic_target - pred_critic_val).squeeze()

        critic_loss=self.critic_criterion(pred_critic_val, critic_target)
        log_probs=torch.log(pred_action_probs + 1e-10)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -torch.mean(action_log_probs * a).detach()
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.step += 1
        if self.step % self.train_log_freq == 0:
            print(f"Step {self.step} | Loss: {total_loss.item():.4f}")