#ifndef TEST_HPP
#define TEST_HPP

#include <cmath>
#include <limits>
#include <algorithm>                 // for std::min
#include "ulmblas_level1.hpp"        // for ulmblas::swap(), ulmblas::geaxpy()
#include "ulmblas_level3.hpp"        // for ulmblas::gemm()

namespace test {

    template <typename TA>
    auto
    norminf(std::size_t m, std::size_t n,
            const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        TA res = 0;
        for (std::size_t i=0; i<m; ++i) {
            TA asum = 0;
            for (std::size_t j=0; j<n; ++j) {
                asum += std::abs(A[i*incRowA + j*incColA]);
            }
            if (std::isnan(asum)) {
                return asum;
            }
            if (asum>res) {
                res = asum;
            }
        }
        return res;
    }

    template <typename TP, typename TA>
    void
    swap(std::size_t m, std::size_t n, const TP *p, std::ptrdiff_t incP,
         std::size_t i0, std::size_t i1,
         TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        std::ptrdiff_t inc = i0<i1 ? 1 : -1;

        i1 += inc;

        for (std::size_t i=i0; i!=i1; i+=inc) {
            if (i!=p[i*incP]) {
                ulmblas::swap(n,
                              &A[i*incRowA], incColA,
                              &A[p[i*incP]*incRowA], incColA);
            }
        }
    }

    template <typename T, typename TP>
    double
    lu_err(std::size_t m, std::size_t n,
           const T *A0, std::ptrdiff_t incRowA0, std::ptrdiff_t incColA0,
           const T *LU, std::ptrdiff_t incRowLU, std::ptrdiff_t incColLU,
           const TP *p, std::ptrdiff_t incP)
    {
        std::size_t k = std::min(m, n);

        T *A = new T[m*n];
        T *L = new T[m*k];
        T *U = new T[k*n];

        // copy L-part from A
        for (std::size_t l=0; l<k; ++l) {
            for (std::size_t i=0; i<m; ++i) {
                L[i+l*m] = (i>l)  ? LU[i*incRowLU + l*incColLU] :
                           (i==l) ? T(1)   :
                           T(0);
            }
        }
        // copy U-part from A
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t l=0; l<k; ++l) {
                U[l + j*k] = (l>j)  ? T(0)
                                    : LU[l*incRowLU + j*incColLU];
            }
        }

        // A = L*U
        ulmblas::gemm(m, n, k,T(1),
                      L, 1, m,
                      U, 1, k,
                      T(0),
                      A, 1, m);

        // A = P^{-1}*A
        swap(m, n, p, incP, std::min(m,n)-1, 0, A, 1, m);

        ulmblas::geaxpy(m, n, -1, A0, incRowA0, incColA0, A, 1, m);

        auto eps = std::numeric_limits<T>::epsilon();

        auto tmp = norminf(m, n, A, 1, m);

        delete [] U;
        delete [] L;
        delete [] A;

        return tmp / (norminf(m, n, A0, incRowA0, incColA0)*eps*std::min(m,n));
    }


} // namespace test

#endif //  TEST_HPP
