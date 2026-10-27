#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include "csr_matrix.h"
#include "../../common/prng.h"

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline long get_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        return usage.ru_maxrss;
#else
        return usage.ru_maxrss * 1024L;
#endif
    }
    return 0;
}

static double compute_csr_checksum(const csr_matrix_t *mat) {
    double sum = 0.0;
    for (size_t i = 0; i < mat->nnz; i++) {
        sum += mat->values[i];
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
    double sparsity = 0.90; // Default: 90% zeros
    uint64_t seed = 42;
    int warmups = 0;
    int repetitions = 1;
    bool output_json = false;
    bool dump_matrix = false;
    int test_mode = 0; // 0: random sparse, 1: identity, 2: zero
    double eps = 1e-12;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--sparsity") == 0 && i + 1 < argc) {
            sparsity = atof(argv[++i]);
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

    if (n == 0 || repetitions <= 0 || sparsity < 0.0 || sparsity >= 1.0) {
        fprintf(stderr, "Invalid parameters: n=%zu, repetitions=%d, sparsity=%.3f\n", n, repetitions, sparsity);
        return 1;
    }

    size_t total_elements = n * n;
    double *dense_A = (double *)malloc(total_elements * sizeof(double));
    double *dense_B = (double *)malloc(total_elements * sizeof(double));

    if (!dense_A || !dense_B) {
        fprintf(stderr, "Dense allocation failed for n=%zu\n", n);
        free(dense_A); free(dense_B);
        return 2;
    }

    if (test_mode == 1) { // Identity: A * I = A
        prng_t prng_a = prng_init(seed);
        prng_fill_sparse_matrix(dense_A, n, sparsity, &prng_a);
        memset(dense_B, 0, total_elements * sizeof(double));
        for (size_t i = 0; i < n; i++) dense_B[i * n + i] = 1.0;
    } else if (test_mode == 2) { // Zero: A * 0 = 0
        prng_t prng_a = prng_init(seed);
        prng_fill_sparse_matrix(dense_A, n, sparsity, &prng_a);
        memset(dense_B, 0, total_elements * sizeof(double));
    } else {
        prng_t prng_a = prng_init(seed);
        prng_fill_sparse_matrix(dense_A, n, sparsity, &prng_a);
        prng_t prng_b = prng_init(seed + 1);
        prng_fill_sparse_matrix(dense_B, n, sparsity, &prng_b);
    }

    // Convert dense inputs to CSR outside multiplication kernel
    double t_conv_start = get_time_sec();
    csr_matrix_t *A = dense_to_csr(dense_A, n, n, eps);
    csr_matrix_t *B = dense_to_csr(dense_B, n, n, eps);
    double t_conv_end = get_time_sec();
    double conv_time_sec = t_conv_end - t_conv_start;

    free(dense_A);
    free(dense_B);

    if (!A || !B) {
        fprintf(stderr, "CSR conversion failed for n=%zu\n", n);
        csr_free(A); csr_free(B);
        return 3;
    }

    // Warm-up runs (discarded from reported samples)
    for (int w = 0; w < warmups; w++) {
        csr_matrix_t *C_warmup = csr_spgemm(A, B, eps);
        csr_free(C_warmup);
    }

    double *times = (double *)malloc(repetitions * sizeof(double));
    csr_matrix_t *C_final = NULL;
    double last_checksum = 0.0;

    for (int r = 0; r < repetitions; r++) {
        double t_start = get_time_sec();
        csr_matrix_t *C = csr_spgemm(A, B, eps);
        double t_end = get_time_sec();
        times[r] = t_end - t_start;

        if (r == repetitions - 1) {
            C_final = C;
            last_checksum = compute_csr_checksum(C_final);
        } else {
            csr_free(C);
        }
    }

    long peak_rss = get_peak_rss_bytes();
    size_t mem_a = csr_memory_bytes(A);
    size_t mem_b = csr_memory_bytes(B);
    size_t mem_c = C_final ? csr_memory_bytes(C_final) : 0;
    size_t total_csr_bytes = mem_a + mem_b + mem_c;

    size_t nnz_a = A->nnz;
    size_t nnz_b = B->nnz;
    size_t nnz_c = C_final ? C_final->nnz : 0;
    double fill_in_ratio = (nnz_a > 0) ? (double)nnz_c / ((double)(nnz_a + nnz_b) * 0.5) : 0.0;
    double output_sparsity = 1.0 - ((double)nnz_c / (double)(n * n));

    if (dump_matrix && n <= 16 && C_final) {
        printf("--- CSR MATRIX C (%zux%zu, nnz=%zu) ---\n", n, n, C_final->nnz);
        double *temp_dense = (double *)malloc(n * n * sizeof(double));
        csr_to_dense(C_final, temp_dense);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                printf("%8.4f ", temp_dense[i * n + j]);
            }
            printf("\n");
        }
        free(temp_dense);
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
        printf("  \"category\": \"sparse_csr\",\n");
        printf("  \"method\": \"gustavson_csr\",\n");
        printf("  \"n\": %zu,\n", n);
        printf("  \"input_sparsity\": %.6f,\n", sparsity);
        printf("  \"output_sparsity\": %.6f,\n", output_sparsity);
        printf("  \"nnz_a\": %zu,\n", nnz_a);
        printf("  \"nnz_b\": %zu,\n", nnz_b);
        printf("  \"nnz_c\": %zu,\n", nnz_c);
        printf("  \"fill_in_ratio\": %.6f,\n", fill_in_ratio);
        printf("  \"seed\": %llu,\n", (unsigned long long)seed);
        printf("  \"repetitions\": %d,\n", repetitions);
        printf("  \"warmups\": %d,\n", warmups);
        printf("  \"kernel_time_median_sec\": %.9f,\n", median_time);
        printf("  \"kernel_time_iqr_sec\": %.9f,\n", iqr);
        printf("  \"conversion_time_sec\": %.9f,\n", conv_time_sec);
        printf("  \"peak_rss_bytes\": %ld,\n", peak_rss);
        printf("  \"csr_storage_bytes\": %zu,\n", total_csr_bytes);
        printf("  \"checksum\": %.10f,\n", last_checksum);
        printf("  \"raw_times\": [");
        for (int r = 0; r < repetitions; r++) {
            printf("%.9f%s", times[r], (r == repetitions - 1) ? "" : ", ");
        }
        printf("]\n");
        printf("}\n");
    } else {
        printf("Method: CSR Gustavson | N: %zu | Sparsity: %.2f%% (Out: %.2f%%) | NNZ(A,B,C): (%zu, %zu, %zu) | Fill-in: %.2fx | Time: %.6f s | ConvTime: %.6f s | CSR Mem: %.2f MB | Peak RSS: %.2f MB | Checksum: %.6f\n",
               n, sparsity * 100.0, output_sparsity * 100.0, nnz_a, nnz_b, nnz_c, fill_in_ratio,
               median_time, conv_time_sec, (double)total_csr_bytes / (1024.0 * 1024.0),
               (double)peak_rss / (1024.0 * 1024.0), last_checksum);
    }

    csr_free(A);
    csr_free(B);
    if (C_final) csr_free(C_final);
    free(times);
    return 0;
}
