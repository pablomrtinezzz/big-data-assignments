#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <omp.h>
#include <arm_neon.h>
#include "../common/prng.h"

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline long get_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        return usage.ru_maxrss; // Darwin reports bytes
#else
        return usage.ru_maxrss * 1024L; // Linux reports kilobytes
#endif
    }
    return 0;
}

/* =========================================================================
 * 1. Sequential Baselines
 * ========================================================================= */

/**
 * Sequential canonical i-j-k multiplication.
 * Stride-n on matrix B, poor spatial cache locality.
 */
void matmul_seq_ijk(const double *restrict A, const double *restrict B, double *restrict C, size_t n) {
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
 * Sequential loop-reordered i-k-j multiplication (Scalar).
 * Contiguous stride-1 access to B and C. Vectorization explicitly disabled
 * to serve as strict scalar baseline.
 */
void matmul_seq_ikj_scalar(const double *restrict A, const double *restrict B, double *restrict C, size_t n) {
    memset(C, 0, n * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double r = A[i * n + k];
            #pragma clang loop vectorize(disable)
            for (size_t j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

/**
 * Sequential loop-reordered i-k-j multiplication with explicit ARM NEON SIMD.
 * 128-bit vector registers (2x float64), unrolled 4x (8 elements per loop iteration).
 */
void matmul_seq_ikj_simd(const double *restrict A, const double *restrict B, double *restrict C, size_t n) {
    memset(C, 0, n * n * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double r = A[i * n + k];
            float64x2_t vr = vdupq_n_f64(r);
            size_t j = 0;
            for (; j + 7 < n; j += 8) {
                float64x2_t vc0 = vld1q_f64(&C[i * n + j]);
                float64x2_t vb0 = vld1q_f64(&B[k * n + j]);
                vc0 = vfmaq_f64(vc0, vr, vb0);
                vst1q_f64(&C[i * n + j], vc0);

                float64x2_t vc1 = vld1q_f64(&C[i * n + j + 2]);
                float64x2_t vb1 = vld1q_f64(&B[k * n + j + 2]);
                vc1 = vfmaq_f64(vc1, vr, vb1);
                vst1q_f64(&C[i * n + j + 2], vc1);

                float64x2_t vc2 = vld1q_f64(&C[i * n + j + 4]);
                float64x2_t vb2 = vld1q_f64(&B[k * n + j + 4]);
                vc2 = vfmaq_f64(vc2, vr, vb2);
                vst1q_f64(&C[i * n + j + 4], vc2);

                float64x2_t vc3 = vld1q_f64(&C[i * n + j + 6]);
                float64x2_t vb3 = vld1q_f64(&B[k * n + j + 6]);
                vc3 = vfmaq_f64(vc3, vr, vb3);
                vst1q_f64(&C[i * n + j + 6], vc3);
            }
            for (; j + 1 < n; j += 2) {
                float64x2_t vc = vld1q_f64(&C[i * n + j]);
                float64x2_t vb = vld1q_f64(&B[k * n + j]);
                vc = vfmaq_f64(vc, vr, vb);
                vst1q_f64(&C[i * n + j], vc);
            }
            for (; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

/* =========================================================================
 * 2. OpenMP Parallel Implementations
 * ========================================================================= */

/**
 * OpenMP parallel canonical i-j-k multiplication.
 */
void matmul_omp_ijk(const double *restrict A, const double *restrict B, double *restrict C, size_t n, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
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
 * OpenMP parallel loop-reordered i-k-j multiplication (Scalar).
 * Supports configurable schedule: static, dynamic, guided.
 */
void matmul_omp_ikj_scalar(const double *restrict A, const double *restrict B, double *restrict C, size_t n,
                           int num_threads, int sched_type, int chunk_size) {
    memset(C, 0, n * n * sizeof(double));
    if (sched_type == 1) { // dynamic
        if (chunk_size > 0) {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    #pragma clang loop vectorize(disable)
                    for (size_t j = 0; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    #pragma clang loop vectorize(disable)
                    for (size_t j = 0; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        }
    } else if (sched_type == 2) { // guided
        #pragma omp parallel for num_threads(num_threads) schedule(guided)
        for (size_t i = 0; i < n; i++) {
            for (size_t k = 0; k < n; k++) {
                double r = A[i * n + k];
                #pragma clang loop vectorize(disable)
                for (size_t j = 0; j < n; j++) {
                    C[i * n + j] += r * B[k * n + j];
                }
            }
        }
    } else { // static (default)
        if (chunk_size > 0) {
            #pragma omp parallel for num_threads(num_threads) schedule(static, chunk_size)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    #pragma clang loop vectorize(disable)
                    for (size_t j = 0; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    #pragma clang loop vectorize(disable)
                    for (size_t j = 0; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        }
    }
}

/**
 * OpenMP parallel loop-reordered i-k-j multiplication with ARM NEON SIMD.
 */
void matmul_omp_ikj_simd(const double *restrict A, const double *restrict B, double *restrict C, size_t n,
                         int num_threads, int sched_type, int chunk_size) {
    memset(C, 0, n * n * sizeof(double));
    if (sched_type == 1) { // dynamic
        if (chunk_size > 0) {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    float64x2_t vr = vdupq_n_f64(r);
                    size_t j = 0;
                    for (; j + 7 < n; j += 8) {
                        float64x2_t vc0 = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb0 = vld1q_f64(&B[k * n + j]);
                        vc0 = vfmaq_f64(vc0, vr, vb0);
                        vst1q_f64(&C[i * n + j], vc0);

                        float64x2_t vc1 = vld1q_f64(&C[i * n + j + 2]);
                        float64x2_t vb1 = vld1q_f64(&B[k * n + j + 2]);
                        vc1 = vfmaq_f64(vc1, vr, vb1);
                        vst1q_f64(&C[i * n + j + 2], vc1);

                        float64x2_t vc2 = vld1q_f64(&C[i * n + j + 4]);
                        float64x2_t vb2 = vld1q_f64(&B[k * n + j + 4]);
                        vc2 = vfmaq_f64(vc2, vr, vb2);
                        vst1q_f64(&C[i * n + j + 4], vc2);

                        float64x2_t vc3 = vld1q_f64(&C[i * n + j + 6]);
                        float64x2_t vb3 = vld1q_f64(&B[k * n + j + 6]);
                        vc3 = vfmaq_f64(vc3, vr, vb3);
                        vst1q_f64(&C[i * n + j + 6], vc3);
                    }
                    for (; j + 1 < n; j += 2) {
                        float64x2_t vc = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb = vld1q_f64(&B[k * n + j]);
                        vc = vfmaq_f64(vc, vr, vb);
                        vst1q_f64(&C[i * n + j], vc);
                    }
                    for (; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    float64x2_t vr = vdupq_n_f64(r);
                    size_t j = 0;
                    for (; j + 7 < n; j += 8) {
                        float64x2_t vc0 = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb0 = vld1q_f64(&B[k * n + j]);
                        vc0 = vfmaq_f64(vc0, vr, vb0);
                        vst1q_f64(&C[i * n + j], vc0);

                        float64x2_t vc1 = vld1q_f64(&C[i * n + j + 2]);
                        float64x2_t vb1 = vld1q_f64(&B[k * n + j + 2]);
                        vc1 = vfmaq_f64(vc1, vr, vb1);
                        vst1q_f64(&C[i * n + j + 2], vc1);

                        float64x2_t vc2 = vld1q_f64(&C[i * n + j + 4]);
                        float64x2_t vb2 = vld1q_f64(&B[k * n + j + 4]);
                        vc2 = vfmaq_f64(vc2, vr, vb2);
                        vst1q_f64(&C[i * n + j + 4], vc2);

                        float64x2_t vc3 = vld1q_f64(&C[i * n + j + 6]);
                        float64x2_t vb3 = vld1q_f64(&B[k * n + j + 6]);
                        vc3 = vfmaq_f64(vc3, vr, vb3);
                        vst1q_f64(&C[i * n + j + 6], vc3);
                    }
                    for (; j + 1 < n; j += 2) {
                        float64x2_t vc = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb = vld1q_f64(&B[k * n + j]);
                        vc = vfmaq_f64(vc, vr, vb);
                        vst1q_f64(&C[i * n + j], vc);
                    }
                    for (; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        }
    } else if (sched_type == 2) { // guided
        #pragma omp parallel for num_threads(num_threads) schedule(guided)
        for (size_t i = 0; i < n; i++) {
            for (size_t k = 0; k < n; k++) {
                double r = A[i * n + k];
                float64x2_t vr = vdupq_n_f64(r);
                size_t j = 0;
                for (; j + 7 < n; j += 8) {
                    float64x2_t vc0 = vld1q_f64(&C[i * n + j]);
                    float64x2_t vb0 = vld1q_f64(&B[k * n + j]);
                    vc0 = vfmaq_f64(vc0, vr, vb0);
                    vst1q_f64(&C[i * n + j], vc0);

                    float64x2_t vc1 = vld1q_f64(&C[i * n + j + 2]);
                    float64x2_t vb1 = vld1q_f64(&B[k * n + j + 2]);
                    vc1 = vfmaq_f64(vc1, vr, vb1);
                    vst1q_f64(&C[i * n + j + 2], vc1);

                    float64x2_t vc2 = vld1q_f64(&C[i * n + j + 4]);
                    float64x2_t vb2 = vld1q_f64(&B[k * n + j + 4]);
                    vc2 = vfmaq_f64(vc2, vr, vb2);
                    vst1q_f64(&C[i * n + j + 4], vc2);

                    float64x2_t vc3 = vld1q_f64(&C[i * n + j + 6]);
                    float64x2_t vb3 = vld1q_f64(&B[k * n + j + 6]);
                    vc3 = vfmaq_f64(vc3, vr, vb3);
                    vst1q_f64(&C[i * n + j + 6], vc3);
                }
                for (; j + 1 < n; j += 2) {
                    float64x2_t vc = vld1q_f64(&C[i * n + j]);
                    float64x2_t vb = vld1q_f64(&B[k * n + j]);
                    vc = vfmaq_f64(vc, vr, vb);
                    vst1q_f64(&C[i * n + j], vc);
                }
                for (; j < n; j++) {
                    C[i * n + j] += r * B[k * n + j];
                }
            }
        }
    } else { // static (default)
        if (chunk_size > 0) {
            #pragma omp parallel for num_threads(num_threads) schedule(static, chunk_size)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    float64x2_t vr = vdupq_n_f64(r);
                    size_t j = 0;
                    for (; j + 7 < n; j += 8) {
                        float64x2_t vc0 = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb0 = vld1q_f64(&B[k * n + j]);
                        vc0 = vfmaq_f64(vc0, vr, vb0);
                        vst1q_f64(&C[i * n + j], vc0);

                        float64x2_t vc1 = vld1q_f64(&C[i * n + j + 2]);
                        float64x2_t vb1 = vld1q_f64(&B[k * n + j + 2]);
                        vc1 = vfmaq_f64(vc1, vr, vb1);
                        vst1q_f64(&C[i * n + j + 2], vc1);

                        float64x2_t vc2 = vld1q_f64(&C[i * n + j + 4]);
                        float64x2_t vb2 = vld1q_f64(&B[k * n + j + 4]);
                        vc2 = vfmaq_f64(vc2, vr, vb2);
                        vst1q_f64(&C[i * n + j + 4], vc2);

                        float64x2_t vc3 = vld1q_f64(&C[i * n + j + 6]);
                        float64x2_t vb3 = vld1q_f64(&B[k * n + j + 6]);
                        vc3 = vfmaq_f64(vc3, vr, vb3);
                        vst1q_f64(&C[i * n + j + 6], vc3);
                    }
                    for (; j + 1 < n; j += 2) {
                        float64x2_t vc = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb = vld1q_f64(&B[k * n + j]);
                        vc = vfmaq_f64(vc, vr, vb);
                        vst1q_f64(&C[i * n + j], vc);
                    }
                    for (; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        } else {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (size_t i = 0; i < n; i++) {
                for (size_t k = 0; k < n; k++) {
                    double r = A[i * n + k];
                    float64x2_t vr = vdupq_n_f64(r);
                    size_t j = 0;
                    for (; j + 7 < n; j += 8) {
                        float64x2_t vc0 = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb0 = vld1q_f64(&B[k * n + j]);
                        vc0 = vfmaq_f64(vc0, vr, vb0);
                        vst1q_f64(&C[i * n + j], vc0);

                        float64x2_t vc1 = vld1q_f64(&C[i * n + j + 2]);
                        float64x2_t vb1 = vld1q_f64(&B[k * n + j + 2]);
                        vc1 = vfmaq_f64(vc1, vr, vb1);
                        vst1q_f64(&C[i * n + j + 2], vc1);

                        float64x2_t vc2 = vld1q_f64(&C[i * n + j + 4]);
                        float64x2_t vb2 = vld1q_f64(&B[k * n + j + 4]);
                        vc2 = vfmaq_f64(vc2, vr, vb2);
                        vst1q_f64(&C[i * n + j + 4], vc2);

                        float64x2_t vc3 = vld1q_f64(&C[i * n + j + 6]);
                        float64x2_t vb3 = vld1q_f64(&B[k * n + j + 6]);
                        vc3 = vfmaq_f64(vc3, vr, vb3);
                        vst1q_f64(&C[i * n + j + 6], vc3);
                    }
                    for (; j + 1 < n; j += 2) {
                        float64x2_t vc = vld1q_f64(&C[i * n + j]);
                        float64x2_t vb = vld1q_f64(&B[k * n + j]);
                        vc = vfmaq_f64(vc, vr, vb);
                        vst1q_f64(&C[i * n + j], vc);
                    }
                    for (; j < n; j++) {
                        C[i * n + j] += r * B[k * n + j];
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * 3. Helper Functions & Driver
 * ========================================================================= */

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

static void run_kernel(const char *method, const double *A, const double *B, double *C, size_t n,
                       int num_threads, int sched_type, int chunk_size) {
    if (strcmp(method, "seq_ikj") == 0) {
        matmul_seq_ikj_scalar(A, B, C, n);
    } else if (strcmp(method, "seq_simd") == 0) {
        matmul_seq_ikj_simd(A, B, C, n);
    } else if (strcmp(method, "omp_ikj") == 0) {
        matmul_omp_ikj_scalar(A, B, C, n, num_threads, sched_type, chunk_size);
    } else if (strcmp(method, "omp_simd") == 0) {
        matmul_omp_ikj_simd(A, B, C, n, num_threads, sched_type, chunk_size);
    } else if (strcmp(method, "seq_ijk") == 0) {
        matmul_seq_ijk(A, B, C, n);
    } else if (strcmp(method, "omp_ijk") == 0) {
        matmul_omp_ijk(A, B, C, n, num_threads);
    } else {
        fprintf(stderr, "Unknown method '%s'\n", method);
        exit(1);
    }
}

int main(int argc, char **argv) {
    char method[32] = "omp_ikj";
    size_t n = 64;
    int num_threads = 1;
    char sched_str[16] = "static";
    int sched_type = 0; // 0: static, 1: dynamic, 2: guided
    int chunk_size = 0;
    uint64_t seed = 42;
    int warmups = 1;
    int repetitions = 5;
    bool output_json = false;
    bool dump_matrix = false;
    int test_mode = 0; // 0: random, 1: identity, 2: zero

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            strncpy(method, argv[++i], sizeof(method) - 1);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) {
            strncpy(sched_str, argv[++i], sizeof(sched_str) - 1);
        } else if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            chunk_size = atoi(argv[++i]);
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

    if (strcmp(sched_str, "dynamic") == 0) sched_type = 1;
    else if (strcmp(sched_str, "guided") == 0) sched_type = 2;
    else sched_type = 0;

    if (n == 0 || repetitions <= 0 || num_threads <= 0) {
        fprintf(stderr, "Invalid parameters: n=%zu, repetitions=%d, threads=%d\n", n, repetitions, num_threads);
        return 1;
    }

    // Force OpenMP runtime thread pool configuration
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    size_t num_elements = n * n;
    size_t bytes_per_matrix = num_elements * sizeof(double);

    double *A = (double *)malloc(bytes_per_matrix);
    double *B = (double *)malloc(bytes_per_matrix);
    double *C = (double *)malloc(bytes_per_matrix);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for n=%zu\n", n);
        free(A); free(B); free(C);
        return 2;
    }

    // Initialize inputs outside the timed region
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
        run_kernel(method, A, B, C, n, num_threads, sched_type, chunk_size);
    }

    // Measured repetitions
    double *times = (double *)malloc(repetitions * sizeof(double));
    double last_checksum = 0.0;

    for (int r = 0; r < repetitions; r++) {
        double t_start = get_time_sec();
        run_kernel(method, A, B, C, n, num_threads, sched_type, chunk_size);
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

    // Calculate median and IQR
    qsort(times, repetitions, sizeof(double), compare_doubles);
    double median_time = (repetitions % 2 == 1) ?
                         times[repetitions / 2] :
                         0.5 * (times[repetitions / 2 - 1] + times[repetitions / 2]);

    double q1 = times[(int)(0.25 * (repetitions - 1))];
    double q3 = times[(int)(0.75 * (repetitions - 1))];
    double iqr = q3 - q1;

    double gflops = (2.0 * (double)n * (double)n * (double)n * 1e-9) / (median_time > 1e-9 ? median_time : 1e-9);

    if (output_json) {
        printf("{\n");
        printf("  \"assignment\": 3,\n");
        printf("  \"method\": \"%s\",\n", method);
        printf("  \"n\": %zu,\n", n);
        printf("  \"threads\": %d,\n", num_threads);
        printf("  \"schedule\": \"%s\",\n", sched_str);
        printf("  \"chunk_size\": %d,\n", chunk_size);
        printf("  \"seed\": %llu,\n", (unsigned long long)seed);
        printf("  \"repetitions\": %d,\n", repetitions);
        printf("  \"warmups\": %d,\n", warmups);
        printf("  \"time_median_sec\": %.9f,\n", median_time);
        printf("  \"time_iqr_sec\": %.9f,\n", iqr);
        printf("  \"gflops\": %.4f,\n", gflops);
        printf("  \"peak_rss_bytes\": %ld,\n", peak_rss);
        printf("  \"checksum\": %.10f,\n", last_checksum);
        printf("  \"raw_times\": [");
        for (int r = 0; r < repetitions; r++) {
            printf("%.9f%s", times[r], (r == repetitions - 1) ? "" : ", ");
        }
        printf("]\n");
        printf("}\n");
    } else {
        printf("Method: %s | N: %zu | Threads: %d | Sched: %s | Median: %.6f s | IQR: %.6f s | GFLOPS: %.2f | RSS: %.2f MB | Checksum: %.6f\n",
               method, n, num_threads, sched_str, median_time, iqr, gflops, (double)peak_rss / (1024.0 * 1024.0), last_checksum);
    }

    free(A);
    free(B);
    free(C);
    free(times);
    return 0;
}
