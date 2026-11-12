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

    def fill_sparse_matrix(self, n: int, sparsity: float):
        """
        Fills an n x n matrix with controlled sparsity in [0.0, 1.0).
        Sparsity is the fraction of zero entries.
        """
        density = 1.0 - sparsity
        mat = []
        for _ in range(n):
            row = []
            for _ in range(n):
                if self.next_double() < density:
                    val = self.next_double()
                    row.append(1e-6 if val == 0.0 else val)
                else:
                    row.append(0.0)
            mat.append(row)
        return mat
