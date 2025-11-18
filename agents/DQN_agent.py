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
        self.target_model.load_state_dict(model.state_dict())
        self.weight_decay=weight_decay
        self.n_actions=self.action_space.n
        self.target_update=target_update
        self.optimizer=model.configure_optimizers()
        self.loss_fn=torch.nn.MSELoss()
        self.step_count=0

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = np.log2(state + 1).astype(np.float32)
        state=np.reshape(state, (1,1,4,4))
        return state

    def action(self, state: np.ndarray) -> int:
        # preprocess state, normalize and turn into tensor
        state_tensor=torch.from_numpy(self.preprocess_state(state))
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
        # learning algorithm using bellman's equation
        # takes in the state, action, reward, and next state and then calculates the 'GROUND TRUTH' values for every item in the batch using the bellman's equation
    
        # preprocess data
        states = np.squeeze(np.stack([self.preprocess_state(t[0]) for t in batch]), axis=1)
        next_states = np.squeeze(np.stack([self.preprocess_state(t[3]) for t in batch]), axis=1)        
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        dones = np.array([t[4] for t in batch])

        # extract the current state and next states (used for model and bellman equation)
        states_tensor = torch.from_numpy(states).float()
        next_states_tensor = torch.from_numpy(next_states).float()

        self.model.eval()
        with torch.no_grad():
            current_qs=self.model(states_tensor).numpy()
            next_qs=self.target_model(next_states_tensor).numpy()

        max_next_qs = np.max(next_qs, axis=1)
        targets = rewards + self.gamma * max_next_qs * (1 - dones)

        target_qs = current_qs.copy()
        target_qs[np.arange(self.batch_size), actions] = targets
        target_qs_tensor = torch.from_numpy(target_qs).float()

        self.model.train()
        preds = self.model(states_tensor)
        loss = self.loss_fn(preds, target_qs_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()
        
        