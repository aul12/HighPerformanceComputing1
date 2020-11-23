/**
 * @file matrix.c
 * @author paul
 * @date 07.11.20
 * Description here TODO
 */
#include "matrix.h"

#include <stdio.h>

#define INDEX(i, j) ((i) * incRow + j * (incCol))

void initMatrix(double *A, size_t m, size_t n, ptrdiff_t incRow, ptrdiff_t incCol) {
    for (size_t i=0; i < m; ++i) {
        for (size_t j=0; j < n; ++j) {
            A[INDEX(i, j)] = (double)(i * n + j + 1);
        }
    }
}

void printMatrix(const double *A, size_t m, size_t n, ptrdiff_t incRow, ptrdiff_t incCol) {
    for (size_t i=0; i < m; ++i) {
        for (size_t j=0; j < n; ++j) {
            printf("%lf\t", A[INDEX(i, j)]);
        }
        printf("\n");
    }
}
