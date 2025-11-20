import time
from typing import Callable
# decorator to count the number of steps per second of the reinforcement learning algorithm (used for benchmarking)

def steps_bench(func: Callable) -> Callable:
    start_time=time.time()
    def wrapper(*args, **kwargs):
        func(* args, **kwargs)
    end_time=time.time()
    pass
