#ifndef PRNG_H
#define PRNG_H

#include <stdint.h>
#include <stddef.h>

/**
 * SplitMix64 pseudo-random number generator.
 * Produces deterministic 64-bit IEEE 754 doubles in [0.0, 1.0).
 * Matches identically across C, Java, and Python implementations.
 */
typedef struct {
    uint64_t state;
} prng_t;

static inline prng_t prng_init(uint64_t seed) {
    prng_t p;
    p.state = seed;
    return p;
}

static inline double prng_next_double(prng_t *p) {
    p->state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = p->state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return (z >> 11) * (1.0 / (1ULL << 53));
}

static inline void prng_fill_matrix(double *mat, size_t n, prng_t *p) {
    size_t total = n * n;
    for (size_t i = 0; i < total; i++) {
        mat[i] = prng_next_double(p);
    }
}

#endif /* PRNG_H */
