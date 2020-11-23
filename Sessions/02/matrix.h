/**
 * @file matrix.h
 * @author paul
 * @date 07.11.20
 * Description here TODO
 */
#ifndef INC_02_MATRIX_H
#define INC_02_MATRIX_H

#include <stddef.h>

void initMatrix(double *A, size_t m, size_t n, ptrdiff_t incRow, ptrdiff_t incCol);

void printMatrix(const double *A, size_t m, size_t n, ptrdiff_t incRow, ptrdiff_t incCol);

#endif //INC_02_MATRIX_H
