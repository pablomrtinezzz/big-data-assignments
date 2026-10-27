#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include "../../common/prng.h"

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline long get_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // macOS Darwin reports ru_maxrss in bytes; Linux in kilobytes.
#ifdef __APPLE__
        return usage.ru_maxrss;
#else
        return usage.ru_maxrss * 1024L;
#endif
    }
    return 0;
}

/**
 * 1. Canonical i-j-k triple-loop multiplication (Baseline).
 * Access pattern in inner loop:
 *   A[i*n + k]: stride-0 (held in accumulator / register)
 *   B[k*n + j]: stride-n (non-contiguous, high cache-miss penalty)
 *   C[i*n + j]: stride-0 during accumulation
 */
void matmul_baseline(const double *restrict A, const double *restrict B, double *restrict C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * 2. Loop Reordering to i-k-j.
 * Access pattern in inner loop:
 *   A[i*n + k]: constant across j (hoisted into register)
 *   B[k*n + j]: stride-1 (contiguous in memory, optimal spatial locality)
 *   C[i*n + j]: stride-1 (contiguous update in memory)
 * Allows vectorization (SIMD NEON on ARM64 / AVX on x86).
 */
void matmul_reordered(const double *restrict A, const double *restrict B, double *restrict C, size_t n) {
    memset(C, 0, n * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (size_t j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

/**
 * 3. Cache Blocking (Tiling) with configurable block size B_tile.
 * Divides n x n matrices into sub-blocks that fit cleanly in the L1/L2 cache.
 * Inner block computation utilizes i-k-j traversal to maximize spatial locality.
 */
void matmul_blocked(const double *restrict A, const double *restrict B, double *restrict C, size_t n, size_t block_size) {
    memset(C, 0, n * n * sizeof(double));
    for (size_t bi = 0; bi < n; bi += block_size) {
        size_t i_max = (bi + block_size < n) ? bi + block_size : n;
        for (size_t bk = 0; bk < n; bk += block_size) {
            size_t k_max = (bk + block_size < n) ? bk + block_size : n;
            for (size_t bj = 0; bj < n; bj += block_size) {
                size_t j_max = (bj + block_size < n) ? bj + block_size : n;
                for (size_t i = bi; i < i_max; i++) {
                    for (size_t k = bk; k < k_max; k++) {
                        double r = A[i * n + k];
                        for (size_t j = bj; j < j_max; j++) {
                            C[i * n + j] += r * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

static double compute_checksum(const double *C, size_t n) {
    double sum = 0.0;
    size_t total = n * n;
    for (size_t i = 0; i < total; i++) {
        sum += C[i];
    }
    return sum;
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

int main(int argc, char **argv) {
    char method[32] = "baseline";
    size_t n = 64;
    size_t block_size = 64;
    uint64_t seed = 42;
    int warmups = 0;
    int repetitions = 1;
    bool output_json = false;
    bool dump_matrix = false;
    int test_mode = 0; // 0: random, 1: identity, 2: zero

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            strncpy(method, argv[++i], sizeof(method) - 1);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            output_json = true;
        } else if (strcmp(argv[i], "--dump") == 0) {
            dump_matrix = true;
        } else if (strcmp(argv[i], "--test-identity") == 0) {
            test_mode = 1;
        } else if (strcmp(argv[i], "--test-zero") == 0) {
            test_mode = 2;
        }
    }

    if (n == 0 || repetitions <= 0 || block_size == 0) {
        fprintf(stderr, "Invalid parameters: n=%zu, repetitions=%d, block_size=%zu\n", n, repetitions, block_size);
        return 1;
    }

    size_t num_elements = n * n;
    size_t bytes_per_matrix = num_elements * sizeof(double);

    double *A = (double *)malloc(bytes_per_matrix);
    double *B = (double *)malloc(bytes_per_matrix);
    double *C = (double *)malloc(bytes_per_matrix);

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed for n=%zu\n", n);
        free(A); free(B); free(C);
        return 2;
    }

    // Input initialization outside timed boundary
    if (test_mode == 1) { // Identity: A * I = A
        prng_t prng_a = prng_init(seed);
        prng_fill_matrix(A, n, &prng_a);
        memset(B, 0, bytes_per_matrix);
        for (size_t i = 0; i < n; i++) B[i * n + i] = 1.0;
    } else if (test_mode == 2) { // Zero: A * 0 = 0
        prng_t prng_a = prng_init(seed);
        prng_fill_matrix(A, n, &prng_a);
        memset(B, 0, bytes_per_matrix);
    } else { // Deterministic pseudo-random inputs
        prng_t prng_a = prng_init(seed);
        prng_fill_matrix(A, n, &prng_a);
        prng_t prng_b = prng_init(seed + 1);
        prng_fill_matrix(B, n, &prng_b);
    }

    // Warm-up runs (discarded from reported samples)
    for (int w = 0; w < warmups; w++) {
        if (strcmp(method, "baseline") == 0) {
            matmul_baseline(A, B, C, n);
        } else if (strcmp(method, "reordered") == 0) {
            matmul_reordered(A, B, C, n);
        } else if (strcmp(method, "blocked") == 0) {
            matmul_blocked(A, B, C, n, block_size);
        } else {
            fprintf(stderr, "Unknown method '%s'. Choose baseline, reordered, or blocked.\n", method);
            free(A); free(B); free(C);
            return 3;
        }
    }

    double *times = (double *)malloc(repetitions * sizeof(double));
    double last_checksum = 0.0;

    for (int r = 0; r < repetitions; r++) {
        double t_start = get_time_sec();
        if (strcmp(method, "baseline") == 0) {
            matmul_baseline(A, B, C, n);
        } else if (strcmp(method, "reordered") == 0) {
            matmul_reordered(A, B, C, n);
        } else if (strcmp(method, "blocked") == 0) {
            matmul_blocked(A, B, C, n, block_size);
        }
        double t_end = get_time_sec();
        times[r] = t_end - t_start;
        last_checksum = compute_checksum(C, n);
    }

    long peak_rss = get_peak_rss_bytes();

    if (dump_matrix && n <= 16) {
        printf("--- MATRIX C (%zux%zu) ---\n", n, n);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                printf("%8.4f ", C[i * n + j]);
            }
            printf("\n");
        }
    }

    qsort(times, repetitions, sizeof(double), compare_doubles);
    double median_time = (repetitions % 2 == 1) ?
                         times[repetitions / 2] :
                         0.5 * (times[repetitions / 2 - 1] + times[repetitions / 2]);

    double q1 = times[(int)(0.25 * (repetitions - 1))];
    double q3 = times[(int)(0.75 * (repetitions - 1))];
    double iqr = q3 - q1;

    if (output_json) {
        printf("{\n");
        printf("  \"assignment\": 2,\n");
        printf("  \"category\": \"dense\",\n");
        printf("  \"method\": \"%s\",\n", method);
        printf("  \"n\": %zu,\n", n);
        printf("  \"block_size\": %zu,\n", block_size);
        printf("  \"seed\": %llu,\n", (unsigned long long)seed);
        printf("  \"repetitions\": %d,\n", repetitions);
        printf("  \"warmups\": %d,\n", warmups);
        printf("  \"time_median_sec\": %.9f,\n", median_time);
        printf("  \"time_iqr_sec\": %.9f,\n", iqr);
        printf("  \"peak_rss_bytes\": %ld,\n", peak_rss);
        printf("  \"checksum\": %.10f,\n", last_checksum);
        printf("  \"raw_times\": [");
        for (int r = 0; r < repetitions; r++) {
            printf("%.9f%s", times[r], (r == repetitions - 1) ? "" : ", ");
        }
        printf("]\n");
        printf("}\n");
    } else {
        printf("Method: %s | N: %zu | Block: %zu | Median Time: %.6f s | IQR: %.6f s | Peak RSS: %.2f MB | Checksum: %.6f\n",
               method, n, block_size, median_time, iqr, (double)peak_rss / (1024.0 * 1024.0), last_checksum);
    }

    free(A);
    free(B);
    free(C);
    free(times);
    return 0;
}
