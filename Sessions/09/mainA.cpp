#include <cassert>
#include <cstddef>
#include <printf.hpp>

namespace tools {
    void
    initMatrix(std::size_t m, std::size_t n,
               double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                A[i * incRowA + j * incColA] = i * n + j + 1;
            }
        }
    }

    void
    printMatrix(std::size_t m, std::size_t n,
                const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                fmt::printf("%6.2lf ", A[i * incRowA + j * incColA]);
            }
            fmt::printf("\n");
        }
        fmt::printf("\n");
    }

    struct DoubleArray {
        explicit DoubleArray(std::size_t n)
                : ptr(new double[n]) {
            assert(ptr);
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

//------------------------------------------------------------------------------

namespace ulmblas {

    void
    pack_A(std::size_t M, std::size_t K,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           std::size_t M_R,
           double *p) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < K; ++j) {
                auto panel = i / M_R;
                auto rowInPanel = i % M_R;
                auto index = panel * K * M_R + j * M_R + rowInPanel;
                p[index] = A[i * incRowA + j * incColA];
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

#ifndef DIM_M
#define DIM_M 9
#endif

#ifndef DIM_K
#define DIM_K 11
#endif

#ifndef COLMAJOR_A
#define COLMAJOR_A 1
#endif


std::size_t M_C = 6;
std::size_t K_C = 4;
std::size_t M_R = 2;


//------------------------------------------------------------------------------

int
main() {
    // Allocate a m x k col major matrix
    std::size_t m = DIM_M;
    std::size_t k = DIM_K;
    tools::DoubleArray A(m * k);
    std::ptrdiff_t incRowA = (COLMAJOR_A) ? 1 : k;
    std::ptrdiff_t incColA = (COLMAJOR_A) ? m : 1;

    // Initialize matrix A
    tools::initMatrix(m, k, A, incRowA, incColA);

    // Print dimensions of A and content of A
    fmt::printf("m = %zu, k = %zu\n", m, k);
    fmt::printf("A = \n");
    tools::printMatrix(m, k, A, incRowA, incColA);

    fmt::printf("M_C = %zu, K_C = %zu, M_R = %zu\n", M_C, K_C, M_R);

    // Allocate a buffer p of size M_C * K_C
    tools::DoubleArray p(M_C * K_C);

    std::size_t m_b = (m + M_C - 1) / M_C;
    std::size_t k_b = (k + K_C - 1) / K_C;

    fmt::printf("A is partitioned into a %zu x %zu block matrix\n", m_b, k_b);

    std::size_t M_ = m % M_C;
    std::size_t K_ = k % K_C;

    for (std::size_t i = 0; i < m_b; ++i) {
        std::size_t M = (i != m_b - 1 || M_ == 0) ? M_C
                                                  : M_;
        for (std::size_t j = 0; j < k_b; ++j) {
            std::size_t K = (j != k_b - 1 || K_ == 0) ? K_C
                                                      : K_;
            fmt::printf("A_{%zu,%zu} is a %zu x %zu matrix\n", i, j, M, K);

            // Print the content of the matrix block A_{i,j}
            fmt::printf("A_{%zu,%zu} = \n", i, j);
            tools::printMatrix(M, K, A + i * M_C * incRowA + j * K_C * incColA,
                               incRowA, incColA);

            // Pack block A_{i,j} in buffer p
            ulmblas::pack_A(M, K, A + i * M_C * incRowA + j * K_C * incColA,
                            incRowA, incColA, M_R, p);

            // Print content of buffer p
            tools::printMatrix(1, M_C * K_C, p, 1, 1);
        }
    }
}
