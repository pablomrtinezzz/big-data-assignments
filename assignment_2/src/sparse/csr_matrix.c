#include "csr_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

csr_matrix_t* csr_create(size_t nrows, size_t ncols, size_t initial_capacity) {
    csr_matrix_t *mat = (csr_matrix_t *)malloc(sizeof(csr_matrix_t));
    if (!mat) return NULL;

    if (initial_capacity == 0) initial_capacity = 16;

    mat->nrows = nrows;
    mat->ncols = ncols;
    mat->nnz = 0;
    mat->capacity = initial_capacity;

    mat->values = (double *)malloc(mat->capacity * sizeof(double));
    mat->col_indices = (size_t *)malloc(mat->capacity * sizeof(size_t));
    mat->row_ptr = (size_t *)calloc(nrows + 1, sizeof(size_t));

    if (!mat->values || !mat->col_indices || !mat->row_ptr) {
        csr_free(mat);
        return NULL;
    }
    return mat;
}

void csr_free(csr_matrix_t *mat) {
    if (!mat) return;
    if (mat->values) free(mat->values);
    if (mat->col_indices) free(mat->col_indices);
    if (mat->row_ptr) free(mat->row_ptr);
    free(mat);
}

static bool csr_ensure_capacity(csr_matrix_t *mat, size_t min_capacity) {
    if (mat->capacity >= min_capacity) return true;
    size_t new_cap = mat->capacity * 2;
    if (new_cap < min_capacity) new_cap = min_capacity;

    double *new_vals = (double *)realloc(mat->values, new_cap * sizeof(double));
    size_t *new_cols = (size_t *)realloc(mat->col_indices, new_cap * sizeof(size_t));
    if (!new_vals || !new_cols) return false;

    mat->values = new_vals;
    mat->col_indices = new_cols;
    mat->capacity = new_cap;
    return true;
}

csr_matrix_t* dense_to_csr(const double *dense, size_t nrows, size_t ncols, double eps) {
    // Pass 1: count non-zeros
    size_t total_nnz = 0;
    for (size_t i = 0; i < nrows * ncols; i++) {
        if (fabs(dense[i]) > eps) {
            total_nnz++;
        }
    }

    csr_matrix_t *mat = csr_create(nrows, ncols, total_nnz > 0 ? total_nnz : 16);
    if (!mat) return NULL;

    size_t idx = 0;
    mat->row_ptr[0] = 0;
    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            double val = dense[i * ncols + j];
            if (fabs(val) > eps) {
                mat->values[idx] = val;
                mat->col_indices[idx] = j;
                idx++;
            }
        }
        mat->row_ptr[i + 1] = idx;
    }
    mat->nnz = idx;
    return mat;
}

void csr_to_dense(const csr_matrix_t *csr, double *dense) {
    memset(dense, 0, csr->nrows * csr->ncols * sizeof(double));
    for (size_t i = 0; i < csr->nrows; i++) {
        size_t start = csr->row_ptr[i];
        size_t end = csr->row_ptr[i + 1];
        for (size_t p = start; p < end; p++) {
            size_t j = csr->col_indices[p];
            dense[i * csr->ncols + j] = csr->values[p];
        }
    }
}

size_t csr_memory_bytes(const csr_matrix_t *mat) {
    if (!mat) return 0;
    size_t bytes = sizeof(csr_matrix_t);
    bytes += mat->capacity * sizeof(double);         // values
    bytes += mat->capacity * sizeof(size_t);         // col_indices
    bytes += (mat->nrows + 1) * sizeof(size_t);      // row_ptr
    return bytes;
}

static int compare_size_t(const void *a, const void *b) {
    size_t sa = *(const size_t *)a;
    size_t sb = *(const size_t *)b;
    if (sa < sb) return -1;
    if (sa > sb) return 1;
    return 0;
}

/**
 * Gustavson's SpGEMM algorithm: C = A * B
 * Operates row-by-row using an accumulator array and marker array for O(1) membership check.
 */
csr_matrix_t* csr_spgemm(const csr_matrix_t *A, const csr_matrix_t *B, double eps) {
    if (!A || !B || A->ncols != B->nrows) return NULL;

    size_t M = A->nrows;
    size_t K = A->ncols;
    size_t N = B->ncols;
    (void)K;

    // Estimate initial capacity based on input NNZs
    size_t est_nnz = A->nnz + B->nnz;
    if (est_nnz < 64) est_nnz = 64;
    csr_matrix_t *C = csr_create(M, N, est_nnz);
    if (!C) return NULL;

    double *accum = (double *)calloc(N, sizeof(double));
    int64_t *marker = (int64_t *)malloc(N * sizeof(int64_t));
    size_t *active_cols = (size_t *)malloc(N * sizeof(size_t));

    if (!accum || !marker || !active_cols) {
        if (accum) free(accum);
        if (marker) free(marker);
        if (active_cols) free(active_cols);
        csr_free(C);
        return NULL;
    }

    for (size_t j = 0; j < N; j++) marker[j] = -1;

    C->row_ptr[0] = 0;
    size_t c_nnz = 0;

    for (size_t i = 0; i < M; i++) {
        size_t num_active = 0;
        size_t a_start = A->row_ptr[i];
        size_t a_end = A->row_ptr[i + 1];

        for (size_t p = a_start; p < a_end; p++) {
            size_t k = A->col_indices[p];
            double a_val = A->values[p];

            size_t b_start = B->row_ptr[k];
            size_t b_end = B->row_ptr[k + 1];

            for (size_t q = b_start; q < b_end; q++) {
                size_t j = B->col_indices[q];
                double b_val = B->values[q];

                if (marker[j] != (int64_t)i) {
                    marker[j] = (int64_t)i;
                    active_cols[num_active++] = j;
                    accum[j] = a_val * b_val;
                } else {
                    accum[j] += a_val * b_val;
                }
            }
        }

        // Sort active column indices to preserve canonical CSR ordering
        if (num_active > 1) {
            qsort(active_cols, num_active, sizeof(size_t), compare_size_t);
        }

        // Ensure C has capacity for newly generated non-zeros
        if (!csr_ensure_capacity(C, c_nnz + num_active)) {
            free(accum); free(marker); free(active_cols);
            csr_free(C);
            return NULL;
        }

        // Write non-zeros to C
        for (size_t idx = 0; idx < num_active; idx++) {
            size_t j = active_cols[idx];
            double val = accum[j];
            if (fabs(val) > eps) {
                C->values[c_nnz] = val;
                C->col_indices[c_nnz] = j;
                c_nnz++;
            }
            accum[j] = 0.0; // Reset accumulator for next row
        }
        C->row_ptr[i + 1] = c_nnz;
    }

    C->nnz = c_nnz;
    free(accum);
    free(marker);
    free(active_cols);
    return C;
}
