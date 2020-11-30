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
        DoubleArray(std::size_t n)
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

#ifndef DIM_K
#define DIM_K 11
#endif

#ifndef DIM_N
#define DIM_N 10
#endif

#ifndef COLMAJOR_B
#define COLMAJOR_B 1
#endif


std::size_t K_C = 4;
std::size_t N_C = 8;
std::size_t N_R = 4;


//------------------------------------------------------------------------------

int
main() {
    // For the moment we will not use the matrix class from session 7
    // 1) Allocate a k x n matrix B
    std::size_t k = DIM_K;
    std::size_t n = DIM_N;
    tools::DoubleArray B(k * n);
    std::ptrdiff_t incRowB = (COLMAJOR_B) ? 1 : n;
    std::ptrdiff_t incColB = (COLMAJOR_B) ? k : 1;

    // 2) Initialize matrix B
    tools::initMatrix(k, n, B, incRowB, incColB);

    // 3) Print dimensions of B and content of B
    fmt::printf("k = %zu, n = %zu\n", k, n);
    fmt::printf("B = \n");
    tools::printMatrix(k, n, B, incRowB, incColB);

    fmt::printf("K_C = %zu, N_C = %zu, N_R = %zu\n", K_C, N_C, N_R);

    // 4) Allocate a buffer p of size K_C * N_C
    tools::DoubleArray p(K_C * N_C);

    std::size_t k_b = (k + K_C - 1) / K_C;
    std::size_t n_b = (n + N_C - 1) / N_C;

    fmt::printf("B is partitioned into a %zu x %zu block matrix\n", k_b, n_b);

    std::size_t K_ = k % K_C;
    std::size_t N_ = n % N_C;

    for (std::size_t i=0; i<k_b; ++i) {
        std::size_t K = (i!=k_b-1 || K_==0) ? K_C
                                            : K_;
        for (std::size_t j=0; j<n_b; ++j) {
            std::size_t N = (j!=n_b-1 || N_==0) ? N_C
                                                : N_;
            fmt::printf("B_{%zu,%zu} is a %zu x %zu matrix\n", i, j, K, N);

            // 5) Print the content of the matrix block B_{i,j}
            fmt::printf("B_{%zu,%zu} = \n", i, j);
            tools::printMatrix(K, N,
                               &B[i*K_C*incRowB + j*N_C*incColB],
                               incRowB, incColB);

            // 6) Pack block B_{i,j} in buffer p
            ulmblas::pack_B(K, N,
                            &B[i*K_C*incRowB + j*N_C*incColB],
                            incRowB, incColB,
                            N_R,
                            p);

            // 7) Print content of buffer p
            tools::printMatrix(1, K_C*N_C, p, 0, 1);
        }
    }
}
