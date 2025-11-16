from pathlib import Path
from collections import deque
from typing import Deque, Tuple, Any
import csv

class ReplayBuffer():
    def __init__(self, file_path: str):
        self.file_path=file_path
        self.replay: Deque[Tuple[Any, Any]]
    
    def add(self, state: Any, value: Any):
        self.replay.append((state, value))

    def save(self):
        if not Path.exists(self.file_path):
            self.file_path.touch()

        with open(self.file_path, 'a', newline='') as csvfile:
            csv_writer=csv.writer(csvfile)
            while self.replay:
                csv_writer.writerow(self.replay.pop())
                
        