"""
Vectorised SplitMix64 generator (NumPy).

Bit-identical to the generator used in Assignments 1-3
(assignment_3/common/prng.h and prng.py): matrix A uses `seed`, matrix B
uses `seed + 1`, values are IEEE-754 doubles in [0, 1), row-major order.
"""
import numpy as np

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def splitmix64_doubles(seed: int, count: int) -> np.ndarray:
    """Return the first `count` doubles produced by SplitMix64(seed)."""
    with np.errstate(over="ignore"):
        idx = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + idx * _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def generate_inputs(n: int, seed: int = 42):
    """Return (A, B) as contiguous n x n float64 arrays, same as the C harness."""
    a = splitmix64_doubles(seed, n * n).reshape(n, n)
    b = splitmix64_doubles(seed + 1, n * n).reshape(n, n)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)
