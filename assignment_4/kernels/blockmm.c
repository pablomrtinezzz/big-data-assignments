/*
 * blockmm.c -- Local block kernel executed inside every Spark task.
 *
 * Reuses the scalar loop-reordered i-k-j kernel validated in Assignment 3
 * (assignment_3/src/matmul_parallel.c, matmul_seq_ikj_scalar), generalised
 * to rectangular blocks and to accumulation (C += A * B) so that the
 * distributed algorithm executes exactly the same inner loop as the
 * sequential and OpenMP baselines. Auto-vectorisation is disabled, as in
 * the Assignment 3 scalar baseline.
 */
#include <stddef.h>

/* C[m x n] += A[m x k] * B[k x n], all row-major and contiguous. */
void block_ikj_acc(const double *restrict A, const double *restrict B,
                   double *restrict C, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t p = 0; p < k; p++) {
            double r = A[i * k + p];
            #pragma clang loop vectorize(disable)
            for (size_t j = 0; j < n; j++) {
                C[i * n + j] += r * B[p * n + j];
            }
        }
    }
}
