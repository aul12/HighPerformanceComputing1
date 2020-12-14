#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <printf.hpp>

#include "test.hpp"
#include "ulmblas_level1.hpp"
#include "ulmblas_level2.hpp"

template<typename TA>
void
initMatrix(std::size_t m, std::size_t n,
           TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           bool withNan = false) {
    // if A is row major initialize A^T
    if (std::abs(incRowA) > std::abs(incColA)) {
        initMatrix(n, m, A, incColA, incRowA, withNan);
        return;
    }
    // if A is col major
    if (withNan) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                A[i * incRowA + j * incColA] = nan("");
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                double rValue = ((double) rand() - RAND_MAX / 2) * 2 / RAND_MAX;
                A[i * incRowA + j * incColA] = rValue;
            }
        }
    }
}

template<typename TA>
void
printMatrix(std::size_t m, std::size_t n,
            const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            fmt::printf("%10.3lf ", A[i * incRowA + j * incColA]);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

//------------------------------------------------------------------------------

template<typename TX>
std::size_t
iamax(std::size_t n, const TX *x, std::ptrdiff_t incX) {
    std::size_t maxElem = 0;
    for (auto c = 1U; c < n; ++c) {
        if (std::abs(x[c * incX]) > std::abs(x[maxElem * incX])) {
            maxElem = c;
        }
    }
    return maxElem;
}

template<typename TA, typename TP>
std::ptrdiff_t
lu_ger(std::size_t m, std::size_t n,
       TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
       TP *p, std::ptrdiff_t incP) {
    auto maxInd = std::min(m, n);
    for (auto j = 0U; j < maxInd; ++j) {
        auto p_j = iamax(m - j, A + j * incRowA + j * incColA, incRowA) + j;
        p[j * incP] = p_j;
        if (p_j != j) {
            ulmblas::swap(n, A + j * incRowA, incColA, A + p_j * incRowA, incColA);
        }

        if (A[j * incRowA + j * incColA] == 0) {
            return j;
        }

        ulmblas::scal(m - j - 1, 1 / A[j * incRowA + j * incColA],
                      A + (j + 1) * incRowA + j * incColA, incRowA);
        ulmblas::ger(m - j - 1, n - j - 1, -1,
                     A + (j + 1) * incRowA + j * incColA, incRowA,
                     A + j * incRowA + (j + 1) * incColA, incColA,
                     A + (j + 1) * incRowA + (j + 1) * incColA, incRowA, incColA);
    }

    return -1;
}

//------------------------------------------------------------------------------


#ifndef COLMAJOR
#define COLMAJOR 1
#endif

int
main() {
    std::size_t m = 500;
    std::size_t n = 500;
    std::ptrdiff_t incRowA = COLMAJOR ? 1 : n;
    std::ptrdiff_t incColA = COLMAJOR ? m : 1;
    std::ptrdiff_t incP = 1;

    auto *A = new double[m * n];
    auto *A0 = new double[m * n];
    auto *p = new std::size_t[std::min(m, n)];

    initMatrix(m, n, A, incRowA, incColA);
    ulmblas::gecopy(m, n, A, incRowA, incColA, A0, incRowA, incColA);

    printMatrix(m, n, A0, incRowA, incColA);

    std::ptrdiff_t info = lu_ger(m, n, A, incRowA, incColA, p, incP);

    printMatrix(m, n, A, incRowA, incColA);
    printMatrix(std::min(m, n), 1, p, incP, 1);
    fmt::printf("info = %d\n", info);

    double err = test::lu_err(m, n,
                              A0, incRowA, incColA,
                              A, incRowA, incColA,
                              p, incP);

    std::printf("err = %e\n", err);

    delete[] A;
    delete[] A0;
    delete[] p;
}

