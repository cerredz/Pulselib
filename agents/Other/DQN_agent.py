import random
from typing import List, Any
from models.tfe import TFELightning
import numpy as np
import torch.optim as optim
import torch
from typing import Tuple

class DQNAgent():
    def __init__(self, action_space: List[Any], model, gamma, weight_decay, epsilon_start, epsilon_end, epsilon_decay, batch_size, target_update):
        self.epsilon=epsilon_start
        self.epsilon_end=epsilon_end
        self.epsilon_decay=epsilon_decay
        self.batch_size=batch_size
        self.action_space=action_space
        self.gamma=gamma
        self.model=model
        self.target_model = TFELightning(lr=model.lr)
        #self.target_model.load_state_dict(model.state_dict())
        self.weight_decay=weight_decay
        self.n_actions=self.action_space.n
        self.target_update=target_update
        self.optimizer=model.configure_optimizers()
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.step_count=0
        self.device = next(model.parameters()).device

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = np.log2(state + 1).astype(np.float32)
        state=np.reshape(state, (1,1,4,4))
        return state

    def action(self, state: np.ndarray) -> int:
        # preprocess state, normalize and turn into tensor
        state_tensor=torch.from_numpy(self.preprocess_state(state)).to(self.device)
        if random.random() < self.epsilon:
            # explore
            action=np.random.randint(0, self.n_actions)
        else:
            # exploit
            self.model.eval()
            with torch.no_grad():
                qs=self.model(state_tensor)
            action = torch.argmax(qs).item()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return action

    def learn(self, batch: List[Tuple]) -> float:
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        states_tensor = torch.tensor(self.preprocess_state_batch(states), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(self.preprocess_state_batch(next_states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device) # Shape (batch, 1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device) # Shape (batch, 1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)     # Shape (batch, 1)

        with torch.no_grad():
            next_qs = self.target_model(next_states_tensor)
            max_next_qs, _ = next_qs.max(dim=1, keepdim=True)
            target_q_values = rewards_tensor + (self.gamma * max_next_qs * (1 - dones_tensor))

        self.model.train()
        current_qs = self.model(states_tensor)
        
        predicted_q_values = current_qs.gather(dim=1, index=actions_tensor)
        loss = self.loss_fn(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    # Helper to vectorize preprocessing
    def preprocess_state_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.log2(np.maximum(states, 1)).astype(np.float32)
        return np.expand_dims(states, axis=1)
        
        