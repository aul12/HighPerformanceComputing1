/**
 * @file quiz01.h
 * @author paul
 * @date 18.11.20
 * Description here TODO
 */
#ifndef INC_01_QUIZ01_H
#define INC_01_QUIZ01_H

#include <stddef.h>

void
dgemv(size_t m, size_t n,
      double alpha,
      const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
      const double *x, ptrdiff_t incX,
      double beta,
      double *y, ptrdiff_t incY);

#endif //INC_01_QUIZ01_H
