import time
from typing import Callable
# decorator to count the number of steps per second of the reinforcement learning algorithm (used for benchmarking)

import time
from typing import Callable
from functools import wraps

_step_counter = 0
_start_time = time.time()
_last_report_time = _start_time
_lock = None 
def steps(reported_every_sec: float = 10.0):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _step_counter, _last_report_time
            _step_counter += 1
            current_time = time.time()
            if current_time - _last_report_time >= reported_every_sec:
                elapsed = current_time - _start_time
                sps = _step_counter / elapsed if elapsed > 0 else 0
                print(f"[SPS] Steps: {_step_counter:,} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Steps/sec: {sps:,.1f}")
                _last_report_time = current_time
            return func(*args, **kwargs)
        return wrapper
    return decorator

def profile(func: Callable) -> dict:
    pass
