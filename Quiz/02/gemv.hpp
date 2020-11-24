#ifndef HPC_GEMV_HPP
#define HPC_GEMV_HPP

#include "matrix.hpp"
#include "vector.hpp"

void gemv(double alpha, const Matrix& A, const Vector& x,
      double beta, Vector& y);

#endif
