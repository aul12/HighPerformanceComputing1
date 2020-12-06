#include <cassert>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <cfloat>
#include <printf.hpp>

//------------------------------------------------------------------------------

void
randMatrix(std::size_t m, std::size_t n,
           double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            A[i * incRowA + j * incColA] = ((double) rand() - RAND_MAX / 2) * 2 / RAND_MAX;
        }
    }
}

void
nanMatrix(std::size_t m, std::size_t n,
          double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            A[i * incRowA + j * incColA] = std::nan("");
        }
    }
}

void
printMatrix(std::size_t m, std::size_t n,
            const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            fmt::printf("%10.3lf ", A[i * incRowA + j * incColA]);
        }
        fmt::printf("\n");
    }
    fmt::printf("\n");
}

double
genorm_inf(std::size_t m, std::size_t n,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
    double res = 0;
    for (std::size_t i = 0; i < m; ++i) {
        double asum = 0;
        for (std::size_t j = 0; j < n; ++j) {
            asum += std::fabs(A[i * incRowA + j * incColA]);
        }
        if (std::isnan(asum)) {
            return asum;
        }
        if (asum > res) {
            res = asum;
        }
    }
    return res;
}

void
geaxpy(std::size_t m, std::size_t n, double alpha,
       const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
       double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB) {
    if (m == 0 || n == 0 || alpha == 0) {
        return;
    }
    // if B is row major:   B^T <- alpha*A^T + B^T
    if (incRowB > incColB) {
        geaxpy(n, m, alpha, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // B is col major:
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            B[i * incRowB + j * incColB] += alpha * A[i * incRowA + j * incColA];
        }
    }
}

void
gecopy(std::size_t m, std::size_t n,
       const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
       double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB) {
    if (m == 0 || n == 0) {
        return;
    }
    // if B is row major:   B^T <- A^T
    if (incRowB > incColB) {
        gecopy(n, m, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // B is col major:
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            B[i * incRowB + j * incColB] = A[i * incRowA + j * incColA];
        }
    }
}

//------------------------------------------------------------------------------

#define MAX(x, y)    ((x)>(y)) ? (x) : (y)

double
gemm_err_est(std::size_t m, std::size_t n, std::size_t k,
             double alpha,
             const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
             const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
             const double *C0, std::ptrdiff_t incRowC0, std::ptrdiff_t incColC0,
             double beta,
             const double *C_, std::ptrdiff_t incRowC_, std::ptrdiff_t incColC_,
             double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC) {
    geaxpy(m, n, -1, C_, incRowC_, incColC_, C, incRowC, incColC);

    double normD = genorm_inf(m, n, C, incRowC, incColC);
    std::size_t N = MAX(m, MAX(n, k));

    if (std::isnan(normD)) {
        return normD;
    }

    if (normD == 0) {
        return 0;
    }

    double normA = 0;
    double normB = 0;

    if (alpha != 0) {
        normB = genorm_inf(k, n, B, incRowB, incColB);
        normA = genorm_inf(m, k, A, incRowA, incColA);
        normA *= fabs(alpha);
    }

    double normC0 = 0;
    if (beta != 0) {
        normC0 = genorm_inf(m, n, C0, incRowC0, incColC0);
        normC0 *= fabs(beta);
    }

    return normD / (DBL_EPSILON * (N * normA * normB + normC0));
}

//------------------------------------------------------------------------------

void
gemm_ref(std::size_t m, std::size_t n, std::size_t k,
         double alpha,
         const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
         const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
         double beta,
         double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC) {
    if (beta != 1) {
        if (beta != 0) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                }
            }
        } else {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    C[i * incRowC + j * incColC] = 0;
                }
            }
        }
    }
    if (k == 0 || alpha == 0) {
        return;
    }
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t l = 0; l < k; ++l) {
            for (std::size_t i = 0; i < m; ++i) {
                C[i * incRowC + j * incColC] += alpha * A[i * incRowA + l * incColA]
                                                * B[l * incRowB + j * incColB];
            }
        }
    }
}

//------------------------------------------------------------------------------

#ifndef DUGEMM_MR_DEFAULT
#define DUGEMM_MR_DEFAULT   4
#endif

#ifndef DUGEMM_NR_DEFAULT
#define DUGEMM_NR_DEFAULT   8
#endif

namespace ulmblas {

    namespace dugemm_parameter {
        std::size_t MR = DUGEMM_MR_DEFAULT;
        std::size_t NR = DUGEMM_NR_DEFAULT;
    } // namespace dugemm_parameter

    void
    ugemm_ref(std::size_t k, double alpha,
              const double *A, const double *B,
              double beta,
              double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC) {
        using namespace dugemm_parameter;

        double AB[MR * NR];

        for (std::size_t i = 0; i < MR * NR; ++i) {
            AB[i] = 0;
        }
        for (std::size_t l = 0; l < k; ++l) {
            for (std::size_t i = 0; i < MR; ++i) {
                for (std::size_t j = 0; j < NR; ++j) {
                    AB[i * NR + j] += A[i] * B[j];
                }
            }
            A += MR;
            B += NR;
        }
        // Yeah, this is unnecessary if (alpha==0). But ok ...
        for (std::size_t i = 0; i < MR * NR; ++i) {
            AB[i] *= alpha;
        }
        // This check for beta is really necessary
        if (beta != 0) {
            for (std::size_t j = 0; j < NR; ++j) {
                for (std::size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                    C[i * incRowC + j * incColC] += AB[i * NR + j];
                }
            }
        } else {
            for (std::size_t j = 0; j < NR; ++j) {
                for (std::size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = AB[i * NR + j];
                }
            }
        }
    }

} // namespace ulmblas

//------------------------------------------------------------------------------

#ifndef ALPHA
#define ALPHA   1
#endif

#ifndef BETA
#define BETA    1
#endif

#ifndef DIM_K
#define DIM_K   128
#endif

#define DIM_MR  DUGEMM_MR_DEFAULT
#define DIM_NR  DUGEMM_NR_DEFAULT

#ifndef COLMAJOR_C
#define COLMAJOR_C 1
#endif

#if BETA == 0
    #define INIT_C nanMatrix
#else
    #define INIT_C randMatrix
#endif

#if COLMAJOR_C
    #define INC_ROW_C 1
    #define INC_COL_C DIM_MR
#else
    #define INC_ROW_C DIM_NR
    #define INC_COL_C 1
#endif

double A[DIM_MR * DIM_K];
double B[DIM_K * DIM_NR];
double C0[DIM_MR * DIM_NR];
double Cref[DIM_MR * DIM_NR];
double Ctest[DIM_MR * DIM_NR];

int
main() {
    randMatrix(DIM_MR, DIM_K, A, 1, DIM_MR);
    randMatrix(DIM_K, DIM_NR, B, DIM_NR, 1);

    INIT_C(DIM_MR, DIM_NR, C0, INC_ROW_C, INC_COL_C);

    gecopy(DIM_MR, DIM_NR, C0, INC_ROW_C, INC_COL_C, Cref, INC_ROW_C, INC_COL_C);
    gemm_ref(DIM_MR, DIM_NR, DIM_K, ALPHA, A, 1, DIM_MR, B, DIM_NR, 1, BETA, Cref, INC_ROW_C, INC_COL_C);

    gecopy(DIM_MR, DIM_NR, C0, INC_ROW_C, INC_COL_C, Ctest, INC_ROW_C, INC_COL_C);
    ulmblas::ugemm_ref(DIM_K, ALPHA, A, B, BETA, Ctest, INC_ROW_C, INC_COL_C);

    auto err = gemm_err_est(DIM_MR, DIM_NR, DIM_K, ALPHA, A, 1, DIM_MR, B, DIM_NR, 1, C0, INC_ROW_C, INC_COL_C, BETA,
                            Ctest, INC_ROW_C, INC_COL_C, Cref, INC_ROW_C, INC_COL_C);

    std::cout << err << std::endl;
}
