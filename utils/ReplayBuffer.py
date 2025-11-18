from pathlib import Path
from collections import deque
from typing import Deque, Tuple, Any
import csv
import random

class ReplayBuffer():
    def __init__(self, file_path: str, capacity: int):
        self.file_path=file_path
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
                
        