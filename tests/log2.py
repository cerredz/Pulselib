import timeit
import math
import numpy as np
from numba import njit

# The number to test
x_float = 12345.6789
x_int = 12345

# Number of executions (we need many to get a readable time)
N = 10_000_000

# --- CANDIDATE 1: math.log2 (The Standard) ---
def test_math_log2():
    return math.log2(x_float)

# --- CANDIDATE 2: math.log(x, 2) (General Log) ---
# This usually involves a division internally: ln(x) / ln(2)
def test_math_log_base():
    return math.log(x_float, 2)

# --- CANDIDATE 3: NumPy on a Scalar ---
# WARNING: This incurs heavy overhead for converting Python float -> NumPy type -> Python float
def test_numpy_scalar():
    return np.log2(x_float)

# --- CANDIDATE 4: Numba (Called from Python) ---
# We define a JIT function, but we call it from Python context.
# There is a small "dispatch" overhead every time you call a JIT function from Python.
@njit
def numba_scalar(val):
    return math.log2(val)

# Warmup Numba
numba_scalar(x_float)

# --- CANDIDATE 5: Integer Bit Length (Integer inputs only) ---
# If you only need the integer part (floor) of the log2 for an int.
def test_bit_length():
    return x_int.bit_length() - 1

print(f"Benchmarking {N:,} iterations for single number operations...\n")

t1 = timeit.timeit(test_math_log2, number=N)
print(f"1. math.log2(x):         {t1:.4f} s  <-- THE WINNER (For Floats)")

t2 = timeit.timeit(test_math_log_base, number=N)
print(f"2. math.log(x, 2):       {t2:.4f} s  (Slower due to division)")

t3 = timeit.timeit(test_numpy_scalar, number=N)
print(f"3. numpy.log2(x):        {t3:.4f} s  (Huge overhead for scalars)")

t4 = timeit.timeit(lambda: numba_scalar(x_float), number=N)
print(f"4. Numba (from Python):  {t4:.4f} s  (Dispatch overhead kills speed)")

t5 = timeit.timeit(test_bit_length, number=N)
print(f"5. x.bit_length() - 1:   {t5:.4f} s  <-- FASTEST (Integers only)")