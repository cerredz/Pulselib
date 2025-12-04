import random
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


class ReplayBufferTorch():
    def __init__(self, device, batch_size, capacity):
        self.device, self.batch_size, self.capacity = device, batch_size, capacity
        self.ptr, self.size, self.storage = 0, 0, None

    def add(self, s, a, r, ns, d):

        if self.storage is None:
            self.storage = [torch.empty((self.capacity, *x.shape[1:]), dtype=x.dtype, device=self.device) for x in [s, a, r, ns, d]]
        
        # Smart circular buffer using modulo (handles wrapping automatically)
        idx = torch.arange(self.ptr, self.ptr + self.batch_size, device=self.device) % self.capacity
        for buf, val in zip(self.storage, [s, a, r, ns, d]): buf[idx] = val
        
        self.ptr = (self.ptr + self.batch_size) % self.capacity
        self.size = min(self.size + self.batch_size, self.capacity)

    def sample(self):
        idx = torch.randint(0, self.size, (self.batch_size,), device=self.device)
        return [buf[idx] for buf in self.storage]