#include <cassert>      // for assert()
#include <cstddef>      // for std::size_t, std::ptrdiff_t
#include <cstdlib>      // for abort()


//------------------------------------------------------------------------------

#ifndef DUGEMM_MR_DEFAULT
#define DUGEMM_MR_DEFAULT   4
#endif

#ifndef DUGEMM_NR_DEFAULT
#define DUGEMM_NR_DEFAULT   64
#endif

#ifndef DGEMM_MC_DEFAULT
#define DGEMM_MC_DEFAULT   256
#endif

#ifndef DGEMM_NC_DEFAULT
#define DGEMM_NC_DEFAULT   2048
#endif

#ifndef DGEMM_KC_DEFAULT
#define DGEMM_KC_DEFAULT   256
#endif

//------------------------------------------------------------------------------
// Tool for handling buffers from session 8
//------------------------------------------------------------------------------

namespace tools {

    class DoubleArray {
        public:
            explicit DoubleArray(std::size_t n) : ptr{new double[n]}, n{n} {};

            DoubleArray(const DoubleArray &) = delete;

            auto operator=(const DoubleArray &) -> DoubleArray& = delete;

            DoubleArray(DoubleArray &&) = delete;

            auto operator=(DoubleArray &&) -> DoubleArray& = delete;

            ~DoubleArray() { delete[] ptr; }

            operator double *() { return ptr; }

        private:
            double *ptr;
            std::size_t n;

    };

} // namespace tools


//------------------------------------------------------------------------------
// BLAS Level 1 functions (for matrices)
//------------------------------------------------------------------------------

namespace ulmblas {

    void
    gescal(std::size_t m, std::size_t n, double alpha,
           double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
        if (alpha == 1) {
            return;
        }

        if (incRowA >= incColA) {
            if (alpha != 0) {
                for (auto i = 0U; i < m; ++i) {
                    for (auto j = 0U; j < n; ++j) {
                        A[i * incRowA + j * incColA] *= alpha;
                    }
                }
            } else {
                for (auto i = 0U; i < m; ++i) {
                    for (auto j = 0U; j < n; ++j) {
                        A[i * incRowA + j * incColA] = 0;
                    }
                }
            }
        } else {
            gescal(n, m, alpha, A, incColA, incRowA);
        }
    }

    void
    geaxpy(std::size_t m, std::size_t n, double alpha,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB) {
        if (alpha == 0) {
            return;
        }

        if (incRowB >= incColB) {
            for (auto i = 0U; i < m; ++i) {
                for (auto j = 0U; j < n; ++j) {
                    B[i * incRowB + j * incColB] += alpha * A[i * incRowA + j * incColA];
                }
            }
        } else {
            geaxpy(n, m, alpha, A, incColA, incRowA, B, incColB, incRowB);
        }
    }

} // namespace ulmblas

//------------------------------------------------------------------------------
// GEMM micro kernel
//------------------------------------------------------------------------------

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
        gescal(MR, NR, beta, C, incRowC, incColC);

        for (auto l = 0U; l < k; ++l) {
            for (auto i = 0U; i < MR; ++i) {
                for (auto j = 0U; j < NR; ++j) {
                    C[incRowC * i + incColC * j] += alpha * A[i + l * MR] * B[l * NR + j];
                }
            }
        }
    }

} // namespace ulmblas

//------------------------------------------------------------------------------
// Packing matrix blocks for GEMM
//------------------------------------------------------------------------------

namespace ulmblas {

    void
    pack_A(std::size_t M, std::size_t K,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           std::size_t M_R,
           double *p) {
        auto maxM = ((M + M_R - 1) / M_R) * M_R;
        for (size_t i = 0; i < maxM; ++i) {
            for (size_t j = 0; j < K; ++j) {
                auto panel = i / M_R;
                auto rowInPanel = i % M_R;
                auto index = panel * K * M_R + j * M_R + rowInPanel;
                if (i < M) {
                    p[index] = A[i * incRowA + j * incColA];
                } else {
                    p[index] = 0;
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
// GEMM macro kernel
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
                auto xAtBorder = (x + 1) * NR > N;
                auto yAtBorder = (y + 1) * MR > M;
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
// GEMM frame algorithm
//------------------------------------------------------------------------------

namespace ulmblas {

    namespace dgemm_parameter {
        std::size_t MC = DGEMM_MC_DEFAULT;
        std::size_t NC = DGEMM_NC_DEFAULT;
        std::size_t KC = DGEMM_KC_DEFAULT;
    } // namespace dgemm_parameter


    void
    gemm(std::size_t m, std::size_t n, std::size_t k,
         double alpha,
         const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
         const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
         double beta,
         double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC) {
        using namespace dugemm_parameter;
        using namespace dgemm_parameter;

        assert(MC % MR == 0);
        assert(NC % NR == 0);

        tools::DoubleArray aTmp{MC * KC};
        tools::DoubleArray bTmp{KC * NC};

        gescal(m, n, beta, C, incRowC, incColC);
        beta = 1;

        if (k != 0) {
            for (auto i = 0U; i < m; i += MC) {
                for (auto j = 0U; j < n; j += NC) {
                    for (auto l = 0U; l < k; l += KC) {
                        const auto mc = (i + MC <= m) ? MC : m % MC;
                        const auto nc = (j + NC <= n) ? NC : n % NC;
                        const auto kc = (l + KC <= k) ? KC : k % KC;

                        pack_A(mc, kc, A + i * incRowA + l * incColA, incRowA, incColA, MR, aTmp);
                        pack_B(kc, nc, B + l * incRowB + j * incColB, incRowB, incColB, NR, bTmp);

                        mgemm(mc, nc, kc, alpha, aTmp, bTmp, beta, C + i * incRowC + j * incColC, incRowC, incColC);
                    }
                }
            }
        } else {
            gescal(m, n, beta, C, incRowC, incColC);
        }

    }

} // namespace ulmblas

#define TRANSPOSED(trans) (*(trans)=='t' or *(trans)=='T')

extern "C" {
    void
    ulm_dgemm_(const char *transA, const char *transB, const int *m, const int *n, const int *k, const double *alpha,
               const double *A, const int *ldA, const double *B, const int *ldB, const double *beta, double *C,
               const int *ldC) {
        auto incRowA = TRANSPOSED(transA) ? *ldA : 1;
        auto incColA = TRANSPOSED(transA) ? 1 : *ldA;
        auto incRowB = TRANSPOSED(transB) ? *ldB : 1;
        auto incColB = TRANSPOSED(transB) ? 1 : *ldB;

        ulmblas::gemm(*m, *n, *k, *alpha, A, incRowA, incColA, B, incRowB, incColB, *beta, C, 1, *ldC);
    }
} // extern "C"

//------------------------------------------------------------------------------

