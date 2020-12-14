#ifndef ULMBLAS_LEVEL2_HPP
#define ULMBLAS_LEVEL2_HPP

#include <cstddef>      // for std::size_t, std::ptrdiff_t
#include <cmath>        // for std::abs (floating point)
#include <cstdlib>      // for std::abs (integer)

namespace ulmblas {

//==============================================================================
//
// BLAS Level 2 functions
//
//==============================================================================

//
//  ger: A <- A + alpha * x*y^T  (rank 1 update)
//
    template<typename Alpha, typename TX, typename TY, typename TA>
    void
    ger(std::size_t m, std::size_t n, Alpha alpha,
        const TX *x, std::ptrdiff_t incX,
        const TY *y, std::ptrdiff_t incY,
        TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
        if (m == 0 || n == 0 || alpha == Alpha(0)) {
            return;
        }
        // if A is row major:   A^T <- A^T + alpha * y*x^T
        if (std::abs(incRowA) > std::abs(incColA)) {
            ger(n, m, alpha, y, incY, x, incX, A, incColA, incRowA);
            return;
        }

        // A is col major:
        for (auto j = 0U; j < n; ++j) {
            for (auto i = 0U; i < m; ++i) {
                A[i * incRowA + j * incColA] += alpha * x[i * incX] * y[j * incY];
            }
        }
    }

} // namespace ulmblas

#endif // ULMBLAS_LEVEL2_HPP
