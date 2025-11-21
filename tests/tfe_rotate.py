import numpy as np
from numba import njit
import time

# ----------------------------------------
# 1. NumPy Built-in
# ----------------------------------------
def numpy_rotate(board):
    return np.rot90(board, 1)

# ----------------------------------------
# 2. Numba Implementation
# ----------------------------------------
@njit
def numba_rotate(board):
    n = board.shape[0]
    m = board.shape[1]
    # We MUST allocate new memory here, which is the slow part
    res = np.zeros((m, n), dtype=np.int32)
    
    for i in range(n):
        for j in range(m):
            res[m - 1 - j, i] = board[i, j]
    return res

# ----------------------------------------
# Benchmark
# ----------------------------------------
board = np.array([[2, 4, 8, 16],
                  [32, 64, 128, 256],
                  [2, 4, 8, 16],
                  [32, 64, 128, 256]], dtype=np.int32)

# Compilation Run
numba_rotate(board)

start = time.perf_counter()
for _ in range(10_000_000):
    numpy_rotate(board)
print(f"NumPy rot90 time: {time.perf_counter() - start:.4f}s")

start = time.perf_counter()
for _ in range(10_000_000):
    numba_rotate(board)
print(f"Numba rotate time: {time.perf_counter() - start:.4f}s")