/**
 * @file main02.cpp.c
 * @author paul
 * @date 04.12.20
 * Description here TODO
 */
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
gescal(std::size_t m, std::size_t n, double alpha,
       double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
    if (m == 0 || n == 0 || alpha == 1) {
        return;
    }
    // A is row major: scale A^T
    if (incRowA > incColA) {
        gescal(n, m, alpha, A, incColA, incRowA);
        return;
    }
    // A is col major:
    if (alpha != 0) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                A[i * incRowA + j * incColA] *= alpha;
            }
        }
    } else {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                A[i * incRowA + j * incColA] = 0;
            }
        }
    }
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

//
// From session 8
//

namespace tools {

    struct DoubleArray {
        DoubleArray(std::size_t n)
                : ptr(new double[n]) {
            if (!ptr) {
                std::abort();
            }
        }

        ~DoubleArray() {
            delete[] ptr;
        }

        operator double *() const {
            return ptr;
        }

        double *const ptr;

    };

} // namespace tools

namespace ulmblas {

    void
    pack_A(std::size_t M, std::size_t K,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           std::size_t M_R,
           double *p) {
        std::size_t m_p = (M + M_R - 1) / M_R;

        if (incRowA < incColA) {
            for (std::size_t J = 0; J < K; ++J) {
                for (std::size_t I = 0; I < M_R * m_p; ++I) {
                    std::size_t mu = M_R * K * (I / M_R) + J * M_R + (I % M_R);

                    p[mu] = (I < M) ? A[I * incRowA + J * incColA]
                                    : 0;
                }
            }
        } else {
            for (std::size_t I = 0; I < M_R * m_p; ++I) {
                for (std::size_t J = 0; J < K; ++J) {
                    std::size_t mu = M_R * K * (I / M_R) + J * M_R + (I % M_R);

                    p[mu] = (I < M) ? A[I * incRowA + J * incColA]
                                    : 0;
                }
            }
        }
    }

    void
    pack_B(std::size_t K, std::size_t N,
           const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
           std::size_t N_R,
           double *p) {
        pack_A(N, K, B, incColB, incRowB, N_R, p);
    }

} // namespace ulmblas

//------------------------------------------------------------------------------

namespace ulmblas {

    void
    mgemm(std::size_t M, std::size_t N, std::size_t K,
          double alpha,
          const double *A, const double *B,
          double beta,
          double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC) {
        using namespace dugemm_parameter;

        const auto boxesY = (M + MR - 1) / MR;
        const auto boxesX = (N + NR - 1) / NR;

        tools::DoubleArray cTmp{NR * MR};

        for (auto y = 0U; y < boxesY; ++y) { // i
            for (auto x = 0U; x < boxesX; ++x) { // j
                auto xAtBorder = (x+1) * NR > N;
                auto yAtBorder = (y+1) * MR > M;
                if (not xAtBorder and not yAtBorder) {
                    ugemm_ref(K, alpha, A + MR * K * y, B + K * NR * x, beta, C + incRowC * MR * y + incColC * NR * x,
                              incRowC, incColC);
                } else {
                    ugemm_ref(K, alpha, A + MR * K * y, B + K * NR * x, 0, cTmp, 1, MR);

                    auto realNR = xAtBorder ? N % NR : NR;
                    auto realMR = yAtBorder ? M % MR : MR;

                    gescal(realMR, realNR, beta, C + incRowC * MR * y + incColC * NR * x, incRowC, incColC);
                    geaxpy(realMR, realNR, 1, cTmp, 1, MR, C + incRowC * MR * y + incColC * NR * x, incRowC, incColC);
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

#ifndef DIM_M
#define DIM_M   101
#endif

#ifndef DIM_N
#define DIM_N   201
#endif

#ifndef DIM_K
#define DIM_K   151
#endif

#define DIM_MR  DUGEMM_MR_DEFAULT
#define DIM_NR  DUGEMM_NR_DEFAULT

#ifndef COLMAJOR_C
#define COLMAJOR_C 1
#endif

#ifndef COLMAJOR_A
#define COLMAJOR_A 1
#endif

#ifndef COLMAJOR_B
#define COLMAJOR_B 1
#endif

int
main() {
    std::size_t incRowC = (COLMAJOR_C) ? 1 : DIM_N;
    std::size_t incColC = (COLMAJOR_C) ? DIM_M : 1;

    std::size_t incRowA = (COLMAJOR_A) ? 1 : DIM_K;
    std::size_t incColA = (COLMAJOR_A) ? DIM_M : 1;

    std::size_t incRowB = (COLMAJOR_B) ? 1 : DIM_N;
    std::size_t incColB = (COLMAJOR_B) ? DIM_K : 1;

    tools::DoubleArray A(DIM_M * DIM_K);
    tools::DoubleArray B(DIM_K * DIM_N);
    tools::DoubleArray C0(DIM_M * DIM_N);
    tools::DoubleArray Cref(DIM_M * DIM_N);
    tools::DoubleArray Ctst(DIM_M * DIM_N);


    if (BETA == 0) {
        nanMatrix(DIM_M, DIM_N, C0, incRowC, incColC);
    } else {
        randMatrix(DIM_M, DIM_N, C0, incRowC, incColC);
    }
    randMatrix(DIM_M, DIM_K, A, incRowA, incColA);
    randMatrix(DIM_K, DIM_N, B, incRowB, incColB);

    gecopy(DIM_M, DIM_N,
           C0, incRowC, incColC,
           Cref, incRowC, incColC);
    gemm_ref(DIM_M, DIM_N, DIM_K,
             ALPHA,
             A, incRowA, incColA,
             B, incRowB, incColB,
             BETA,
             Cref, incRowC, incColC);

    gecopy(DIM_M, DIM_N,
           C0, incRowC, incColC,
           Ctst, incRowC, incColC);

    std::size_t Mz = (DIM_M + DIM_MR - 1) / DIM_MR * DIM_MR;
    std::size_t Nz = (DIM_N + DIM_NR - 1) / DIM_NR * DIM_NR;

    tools::DoubleArray A_(Mz * DIM_K);
    tools::DoubleArray B_(DIM_K * Nz);

    ulmblas::pack_A(DIM_M, DIM_K, A, incRowA, incColA, DIM_MR, A_);
    ulmblas::pack_B(DIM_K, DIM_N, B, incRowB, incColB, DIM_NR, B_);

    ulmblas::mgemm(DIM_M, DIM_N, DIM_K,
                   ALPHA,
                   A_, B_,
                   BETA,
                   Ctst, incRowC, incColC);

    double err = gemm_err_est(DIM_M, DIM_N, DIM_K,
                              ALPHA,
                              A, incRowA, incColA,
                              B, incRowB, incColB,
                              C0, incRowC, incColC,
                              BETA,
                              Cref, incRowC, incColC,
                              Ctst, incRowC, incColC);

    fmt::printf("%7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %12s\n",
                "MR", "NR", "k", "incRowC", "incColC",
                "incRowA", "incColA", "incRowB", "incColB",
                "alpha", "beta", "error");
    fmt::printf("%7zu %7zu %7zu %7zu %7zu %7zu %7zu %7zu %7zu "
                "%7.2lf %7.2lf %12.2e\n",
                DIM_M, DIM_N, DIM_K, incRowC, incColC, incRowA, incColA,
                incRowB, incColB, ALPHA, BETA, err);
}
