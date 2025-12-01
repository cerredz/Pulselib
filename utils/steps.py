import time
from typing import Callable
# decorator to count the number of steps per second of the reinforcement learning algorithm (used for benchmarking)
from rich import print as r
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

def profile(f):
    @wraps(f)
    def wrapper(*a, **kw):
        s = time.perf_counter()
        try:
            res = f(*a, **kw)
        except Exception as e:
            r(f"[red]Failed[/] [bold]{f.__name__}[/] raised {type(e).__name__} after "
              f"[bold]{(time.perf_counter()-s)*1000:,.1f}ms[/]")
            raise
        ms = (time.perf_counter() - s) * 1000
        color = "green" if ms < 10 else "yellow" if ms < 100 else "red"
        r(f"[cyan]Profile[/] [white]{f.__name__}[/] {a or ''}{kw or ''} → {res!r}  "
          f"[{color}]{ms:8.2f} ms[/]")
        return res
    return wrapper
