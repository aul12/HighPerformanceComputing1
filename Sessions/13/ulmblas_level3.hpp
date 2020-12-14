#ifndef ULMBLAS_LEVEL3_HPP
#define ULMBLAS_LEVEL3_HPP

#include <cstddef>              // for std::size_t, std::ptrdiff_t
#include <cmath>                // for std::abs (floating point)
#include <cstdlib>              // for std::abs (integer)

#include "ulmblas_level1.hpp"   // for gescal

namespace ulmblas {

//==============================================================================
//
// BLAS Level 3 functions
//
//==============================================================================

//
//  gemm: C <- beta*C + alpha * A * B
//
//  Note: This is just a simple reference implementation!
//
    template <typename Alpha, typename TA, typename TB, typename Beta, typename TC>
    void
    gemm(std::size_t m, std::size_t n, std::size_t k,
         Alpha alpha,
         const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
         const TB *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
         Beta beta,
         TC *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
    {
        gescal(m, n, beta, C, incRowC, incColC);
        if (m==0 || n==0 || k==0 || alpha==Alpha(0)) {
            return;
        }
        for (std::size_t l=0; l<k; ++l) {
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    C[i*incRowC+j*incColC] += alpha*A[i*incRowA+l*incColA]
                                              *B[l*incRowB+j*incColB];
                }
            }
        }
    }

} // namespace ulmblas

#endif // ULMBLAS_LEVEL2_HPP
