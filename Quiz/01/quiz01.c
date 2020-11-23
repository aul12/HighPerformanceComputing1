/**
 * @file quiz01.c
 * @author paul
 * @date 18.11.20
 * Description here TODO
 */

#ifdef USE_AS_LIB

#include "quiz01.h"

#else
#include <stddef.h>
#endif

#define ABS(x) ((x) < 0 ? -(x) : (x))

void
dgemv(size_t m, size_t n,
      double alpha,
      const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
      const double *
      x, ptrdiff_t incX,
      double beta,
      double *
      y, ptrdiff_t incY) {
    if (beta == 0) {
        for (size_t c = 0; c < m; ++c) {
            y[c * incY] = 0;
        }
    } else if (beta != 1) {
        for (size_t c = 0; c < m; ++c) {
            y[c * incY] *= beta;
        }
    }

    if (alpha != 0) {
        if (ABS(incRowA) > ABS(incColA)) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    y[i * incY] += A[i * incRowA + j * incColA] * x[j * incX] * alpha;
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < m; ++i) {
                    y[i * incY] += A[i * incRowA + j * incColA] * x[j * incX] * alpha;
                }
            }
        }
    }
}
