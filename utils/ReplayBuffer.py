import random
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