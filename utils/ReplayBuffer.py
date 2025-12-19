import random
from turtle import position
import torch

class ReplayBuffer():
    def __init__(self, file_path: str, capacity: int):
        self.capacity = capacity
        self.buffer = [] # Use a list for O(1) random access
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Overwrite the old data (Circular Buffer)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# pytorch replay buffer class
## (right now buffer lives in RAM, going to look into how to move to disk to store even bigger replays)

class ReplayBufferTorch:
    def __init__(self, file_path: str, capacity: int, device: str, state_dim: int, action_dim: int):
        assert state_dim > 0 and action_dim > 0

        self.capacity=capacity
        self.device=device
        self.state_dim=state_dim
        self.action_dim=action_dim

        self.state_buffer=torch.zeros((capacity, self.state_dim), device=self.device, dtype=torch.int32)
        self.action_buffer=torch.zeros((capacity, self.action_dim), device=self.device, dtype=torch.int32)
        self.next_state_buffer=torch.zeros((capacity, self.state_dim), device=self.device, dtype=torch.int32)
        self.reward_buffer=torch.zeros(capacity, device=self.device, dtype=torch.int32)
        self.terminated_buffer=torch.zeros(capacity, device=self.device, dtype=torch.int32)

        self.position=0
        self.size=0
        self.full_buffer=False # bool flag to determine if we had to reset the position back to 0 after filling up the buffer

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):
        batch_size=state.shape[0]

        idx = torch.arange(self.position, self.position + batch_size, device=self.device) % self.capacity        
        
        self.state_buffer[idx]=state
        self.action_buffer[idx]=action
        self.reward_buffer[idx] = reward
        self.next_state_buffer[idx]=next_state
        self.terminated_buffer[idx]=done

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int):
        idx=torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.state_buffer[idx],
            self.action_buffer[idx],
            self.reward_buffer[idx],
            self.next_state_buffer[idx],
            self.terminated_buffer[idx]
        )