#ifndef ULMBLAS_LEVEL1_HPP
#define ULMBLAS_LEVEL1_HPP

#include <cstddef>      // for std::size_t, std::ptrdiff_t
#include <cmath>        // for std::abs (floating point)
#include <cstdlib>      // for std::abs (integer)

namespace ulmblas {

//==============================================================================
//
// BLAS Level 1 functions for vectors
//
//==============================================================================

//
// copy:  y <- x
//
// (adopted from session 3)
//
    template <typename TX, typename TY>
    void
    copy(std::size_t n, const TX *x, std::ptrdiff_t incX,
         TY *y, std::ptrdiff_t incY)
    {
        for (std::size_t i=0; i<n; ++i) {
            y[i*incY] = x[i*incX];
        }
    }

//
// swap:  x <-> y
//
// (adopted from session 3)
//
    template <typename TX, typename TY>
    void
    swap(std::size_t n, TX *x, std::ptrdiff_t incX, TY *y, std::ptrdiff_t incY)
    {
        for (std::size_t i=0; i<n; ++i) {
            TX t = x[i*incX];
            x[i*incX] = y[i*incY];
            y[i*incY] = t;
        }
    }

//
// scal: x <- alpha*x
//
// (adopted from session 6)
//
    template <typename Alpha, typename TX>
    void
    scal(std::size_t n, Alpha alpha, TX *x, std::ptrdiff_t incX)
    {
        if (alpha==Alpha(1)) {
            return;
        }
        if (alpha==Alpha(0)) {
            for (std::size_t i=0; i<n; ++i) {
                x[i*incX] = 0;
            }
        } else {
            for (std::size_t i=0; i<n; ++i) {
                x[i*incX] *= alpha;
            }
        }
    }

//
// axpy: y <- y + alpha*x
//
// (adopted from session 6)
//
    template <typename Alpha, typename TX, typename TY>
    void
    axpy(std::size_t n, Alpha alpha,
         const TX *x, std::ptrdiff_t incX,
         TY *y, std::ptrdiff_t incY)
    {
        if (alpha==Alpha(0)) {
            return;
        }
        for (std::size_t i=0; i<n; ++i) {
            y[i*incY] += alpha*x[i*incX];
        }
    }

//==============================================================================
//
// BLAS Level 1 functions for matrices
//
//==============================================================================

//
// gecopy: B <- A
//
// (adopted from session 6)
//
    template <typename TA, typename TB>
    void
    gecopy(std::size_t m, std::size_t n,
           const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           TB *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
    {
        if (m==0 || n==0) {
            return;
        }
        // if B is row major:   B^T <- A^T
        if (std::abs(incRowB) > std::abs(incColB)) {
            gecopy(n, m, A, incColA, incRowA, B, incColB, incRowB);
            return;
        }
        // B is col major:
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] = A[i*incRowA+j*incColA];
            }
        }
    }

//
// gescal: A <- alpha*A
//
// (adopted from session 6)
//
    template <typename Alpha, typename TA>
    void
    gescal(std::size_t m, std::size_t n, Alpha alpha,
           TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        if (alpha==Alpha(1) || m==0 || n==0) {
            return;
        }
        // if A is row major: A^T <- alpha*A^T
        if (std::abs(incRowA) > std::abs(incColA)) {
            gescal(n, m, alpha, A, incColA, incRowA);
            return;
        }
        // A is col major:
        if (alpha!=Alpha(0)) {
            // Scale A column wise with alpha
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    A[i*incRowA+j*incColA] *= alpha;
                }
            }
        } else {
            // A might contain Nan values: overwrite with zeros
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    A[i*incRowA+j*incColA] = 0;
                }
            }
        }
    }

//
// geaxpy: B <- B + alpha*A
//
// (adopted from session 6)
//
    template <typename Alpha, typename TA, typename TB>
    void
    geaxpy(std::size_t m, std::size_t n, Alpha alpha,
           const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           TB *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
    {
        if (m==0 || n==0) {
            return;
        }
        // if B is row major:   B^T <- alpha*A^T + B^T
        if (std::abs(incRowB) > std::abs(incColB)) {
            geaxpy(n, m, alpha, A, incColA, incRowA, B, incColB, incRowB);
            return;
        }
        // B is col major:
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] += alpha*A[i*incRowA+j*incColA];
            }
        }
    }

} // namespace ulmblas

#endif // ULMBLAS_LEVEL1_HPP
