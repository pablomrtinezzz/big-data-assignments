package common;

/**
 * SplitMix64 pseudo-random number generator.
 * Produces deterministic 64-bit IEEE 754 doubles in [0.0, 1.0).
 * Matches identically across C, Java, and Python implementations.
 */
public class PRNG {
    private long state;

    public PRNG(long seed) {
        this.state = seed;
    }

    public double nextDouble() {
        state += 0x9e3779b97f4a7c15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z = z ^ (z >>> 31);
        return (z >>> 11) * (1.0 / (1L << 53));
    }

    public void fillMatrix(double[][] mat, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mat[i][j] = nextDouble();
            }
        }
    }
}
