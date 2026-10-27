"""
SplitMix64 pseudo-random number generator.
Produces deterministic 64-bit IEEE 754 doubles in [0.0, 1.0).
Matches identically across C, Java, and Python implementations.
"""

class PRNG:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_double(self) -> float:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return (z >> 11) * (1.0 / (1 << 53))

    def fill_matrix(self, n: int):
        mat = []
        for _ in range(n):
            row = [self.next_double() for _ in range(n)]
            mat.append(row)
        return mat
