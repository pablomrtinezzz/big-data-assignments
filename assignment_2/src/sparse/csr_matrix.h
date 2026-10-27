#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/**
 * Compressed Sparse Row (CSR) matrix representation.
 * - values: non-zero floating point entries (length: nnz)
 * - col_indices: column indices corresponding to each value (length: nnz)
 * - row_ptr: index boundaries for rows (length: nrows + 1)
 */
typedef struct {
    size_t nrows;
    size_t ncols;
    size_t nnz;
    size_t capacity;
    double *values;
    size_t *col_indices;
    size_t *row_ptr;
} csr_matrix_t;

csr_matrix_t* csr_create(size_t nrows, size_t ncols, size_t initial_capacity);
void csr_free(csr_matrix_t *mat);
csr_matrix_t* dense_to_csr(const double *dense, size_t nrows, size_t ncols, double eps);
void csr_to_dense(const csr_matrix_t *csr, double *dense);
size_t csr_memory_bytes(const csr_matrix_t *mat);

/**
 * Gustavson's Sparse General Matrix Multiplication (SpGEMM): C = A * B.
 * Returns newly allocated CSR matrix C.
 */
csr_matrix_t* csr_spgemm(const csr_matrix_t *A, const csr_matrix_t *B, double eps);

#endif /* CSR_MATRIX_H */
