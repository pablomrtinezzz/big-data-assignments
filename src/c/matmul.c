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
        // On macOS Darwin, ru_maxrss is reported in bytes.
        return usage.ru_maxrss;
    }
    return 0;
}

/**
 * Standard canonical triple-loop dense matrix multiplication (i-j-k).
 * Contiguous row-major storage: A[i * n + k], B[k * n + j], C[i * n + j].
 * Arithmetic work: exactly 2 * n^3 operations (n^3 multiplications, n^3 additions).
 */
void matmul_ijk(const double *A, const double *B, double *C, size_t n) {
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
    size_t n = 64;
    uint64_t seed = 42;
    int warmups = 0;
    int repetitions = 1;
    bool output_json = false;
    bool print_matrices = false;
    int test_mode = 0; // 0: random, 1: identity, 2: zero

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            output_json = true;
        } else if (strcmp(argv[i], "--print") == 0) {
            print_matrices = true;
        } else if (strcmp(argv[i], "--test-identity") == 0) {
            test_mode = 1;
        } else if (strcmp(argv[i], "--test-zero") == 0) {
            test_mode = 2;
        }
    }

    if (n == 0 || repetitions <= 0) {
        fprintf(stderr, "Invalid parameters: n=%zu, repetitions=%d\n", n, repetitions);
        return 1;
    }

    size_t num_elements = n * n;
    size_t bytes_per_matrix = num_elements * sizeof(double);
    size_t total_payload_bytes = 3 * bytes_per_matrix;

    double *A = (double *)malloc(bytes_per_matrix);
    double *B = (double *)malloc(bytes_per_matrix);
    double *C = (double *)malloc(bytes_per_matrix);

    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate memory for %zux%zu matrices\n", n, n);
        free(A); free(B); free(C);
        return 2;
    }

    // Populate matrices based on mode
    if (test_mode == 1) { // Identity: A random, B = Identity
        prng_t prng = prng_init(seed);
        prng_fill_matrix(A, n, &prng);
        memset(B, 0, bytes_per_matrix);
        for (size_t i = 0; i < n; i++) B[i * n + i] = 1.0;
    } else if (test_mode == 2) { // Zero: A random, B = Zero
        prng_t prng = prng_init(seed);
        prng_fill_matrix(A, n, &prng);
        memset(B, 0, bytes_per_matrix);
    } else { // Standard random test
        prng_t prng = prng_init(seed);
        prng_fill_matrix(A, n, &prng);
        prng_fill_matrix(B, n, &prng);
    }
    memset(C, 0, bytes_per_matrix);

    // Warm-up runs (not timed/reported)
    for (int w = 0; w < warmups; w++) {
        matmul_ijk(A, B, C, n);
    }

    // Timed repetitions
    double *times = (double *)malloc(sizeof(double) * repetitions);
    double last_checksum = 0.0;

    for (int r = 0; r < repetitions; r++) {
        // Zero output matrix before timed run
        memset(C, 0, bytes_per_matrix);

        double t0 = get_time_sec();
        matmul_ijk(A, B, C, n);
        double t1 = get_time_sec();

        times[r] = t1 - t0;
        // Compute checksum as sink to prevent dead-code elimination
        last_checksum = compute_checksum(C, n);
    }

    // Compute statistics
    double *sorted_times = (double *)malloc(sizeof(double) * repetitions);
    memcpy(sorted_times, times, sizeof(double) * repetitions);
    qsort(sorted_times, repetitions, sizeof(double), compare_doubles);

    double median_time;
    if (repetitions % 2 == 1) {
        median_time = sorted_times[repetitions / 2];
    } else {
        median_time = 0.5 * (sorted_times[repetitions / 2 - 1] + sorted_times[repetitions / 2]);
    }

    double q1, q3, iqr;
    if (repetitions == 1) {
        q1 = sorted_times[0];
        q3 = sorted_times[0];
        iqr = 0.0;
    } else {
        int mid = repetitions / 2;
        if (mid % 2 == 1) {
            q1 = sorted_times[mid / 2];
            q3 = sorted_times[repetitions - 1 - (mid / 2)];
        } else {
            q1 = 0.5 * (sorted_times[mid / 2 - 1] + sorted_times[mid / 2]);
            q3 = 0.5 * (sorted_times[repetitions - mid + mid / 2 - 1] + sorted_times[repetitions - mid + mid / 2]);
        }
        iqr = q3 - q1;
    }

    double min_time = sorted_times[0];
    double max_time = sorted_times[repetitions - 1];
    long peak_rss = get_peak_rss_bytes();

    if (print_matrices && n <= 10) {
        printf("Result matrix C (%zux%zu):\n", n, n);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                printf("%10.6f ", C[i * n + j]);
            }
            printf("\n");
        }
    }

    if (output_json) {
        printf("{\n");
        printf("  \"language\": \"c\",\n");
        printf("  \"n\": %zu,\n", n);
        printf("  \"seed\": %llu,\n", (unsigned long long)seed);
        printf("  \"warmups\": %d,\n", warmups);
        printf("  \"repetitions\": %d,\n", repetitions);
        printf("  \"times_sec\": [");
        for (int r = 0; r < repetitions; r++) {
            printf("%.9f%s", times[r], (r == repetitions - 1) ? "" : ", ");
        }
        printf("],\n");
        printf("  \"median_time_sec\": %.9f,\n", median_time);
        printf("  \"iqr_sec\": %.9f,\n", iqr);
        printf("  \"min_time_sec\": %.9f,\n", min_time);
        printf("  \"max_time_sec\": %.9f,\n", max_time);
        printf("  \"payload_bytes\": %zu,\n", total_payload_bytes);
        printf("  \"peak_rss_bytes\": %ld,\n", peak_rss);
        printf("  \"checksum\": %.12e\n", last_checksum);
        printf("}\n");
    } else {
        printf("C (n=%zu): Repetitions=%d, Median=%.6f s, IQR=%.6f s, Min=%.6f s, Max=%.6f s, Peak RSS=%.2f MB, Checksum=%.6f\n",
               n, repetitions, median_time, iqr, min_time, max_time, (double)peak_rss / (1024.0 * 1024.0), last_checksum);
    }

    free(A);
    free(B);
    free(C);
    free(times);
    free(sorted_times);
    return 0;
}
