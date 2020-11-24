#include <cassert>
#include "gemv.hpp"
#include "matrix.hpp"
#include "vector.hpp"

// y \leftarrow \beta y + \alpha A x
void gemv(double alpha, const Matrix& A, const Vector& x,
      double beta, Vector& y) {
    assert(A.n == x.n and A.m == y.n);

    if (beta == 0) {
        for (std::size_t c=0; c<y.n; ++c) {
            y(c) = 0;
        }
    } else if (beta != 1) {
        for (std::size_t c=0; c<y.n; ++c) {
            y(c) *= beta;
        }
    }

    if (alpha != 0) {
        for (std::size_t m_i=0; m_i<A.m; ++m_i) {
            for (std::size_t n_i=0; n_i<A.n; ++n_i) {
                y(m_i) += A(m_i, n_i) * x(n_i) * alpha;
            }
        }
    }
}
