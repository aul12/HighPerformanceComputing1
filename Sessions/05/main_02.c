#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <sys/times.h>
#include <unistd.h>

//------------------------------------------------------------------------------

#define MY_ABS(x)   ((x)<0 ? -(x) : (x))

//------------------------------------------------------------------------------

void
initMatrix(size_t m, size_t n, double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
           bool withNan) {
    // if A is row major initialize A^T
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
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

void
printMatrix(size_t m, size_t n,
            const double *A, ptrdiff_t incRowA, ptrdiff_t incColA) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%10.3lf ", A[i * incRowA + j * incColA]);
        }
        printf("\n");
    }
    printf("\n");
}

//------------------------------------------------------------------------------

void
dgeaxpy(size_t m, size_t n, double alpha,
        const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
        double *B, ptrdiff_t incRowB, ptrdiff_t incColB) {
    if (m == 0 || n == 0) {
        return;
    }
    // if B is row major:   B^T <- alpha*A^T + B^T
    if (MY_ABS(incRowB) > MY_ABS(incColB)) {
        dgeaxpy(n, m, alpha, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // B is col major:
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            B[i * incRowB + j * incColB] += alpha * A[i * incRowA + j * incColA];
        }
    }
}

void
dgecopy(size_t m, size_t n,
        const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
        double *B, ptrdiff_t incRowB, ptrdiff_t incColB) {
    if (m == 0 || n == 0) {
        return;
    }
    // if B is row major:   B^T <- A^T
    if (MY_ABS(incRowB) > MY_ABS(incColB)) {
        dgecopy(n, m, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // B is col major:
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            B[i * incRowB + j * incColB] = A[i * incRowA + j * incColA];
        }
    }
}

void
dgescal(size_t m, size_t n, double alpha,
        double *A, ptrdiff_t incRowA, ptrdiff_t incColA) {
    if (alpha == 1 || m == 0 || n == 0) {
        return;
    }
    // if A is row major: A^T <- alpha*A^T
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
        dgescal(n, m, alpha, A, incColA, incRowA);
        return;
    }
    // A is col major:
    if (alpha != 0) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                A[i * incRowA + j * incColA] *= alpha;
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                A[i * incRowA + j * incColA] = 0;
            }
        }
    }
}

// This operation is not cache friendly!
double
dgenorm_inf(size_t m, size_t n,
            const double *A, ptrdiff_t incRowA, ptrdiff_t incColA) {
    double res = 0;
    for (size_t i = 0; i < m; ++i) {
        double asum = 0;
        for (size_t j = 0; j < n; ++j) {
            asum += fabs(A[i * incRowA + j * incColA]);
        }
        if (asum > res) {
            res = asum;
        }
    }
    return res;
}

//------------------------------------------------------------------------------

void
dgemm_ijl(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgescal(m, n, beta, C, incRowC, incColC);
    if (m == 0 || n == 0 || k == 0 || alpha == 0) {
        return;
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < k; ++l) {
                C[i * incRowC + j * incColC] +=
                        alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}

void
dgemm_ilj(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgescal(m, n, beta, C, incRowC, incColC);
    if (m == 0 || n == 0 || k == 0 || alpha == 0) {
        return;
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < k; ++l) {
            for (size_t j = 0; j < n; ++j) {
                C[i * incRowC + j * incColC] +=
                        alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}

void
dgemm_lij(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgescal(m, n, beta, C, incRowC, incColC);
    if (m == 0 || n == 0 || k == 0 || alpha == 0) {
        return;
    }

    for (size_t l = 0; l < k; ++l) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C[i * incRowC + j * incColC] +=
                        alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}

void
dgemm_lji(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgescal(m, n, beta, C, incRowC, incColC);
    if (m == 0 || n == 0 || k == 0 || alpha == 0) {
        return;
    }

    for (size_t l = 0; l < k; ++l) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                C[i * incRowC + j * incColC] +=
                        alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}

void
dgemm_jli(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgescal(m, n, beta, C, incRowC, incColC);
    if (m == 0 || n == 0 || k == 0 || alpha == 0) {
        return;
    }

    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            for (size_t i = 0; i < m; ++i) {
                C[i * incRowC + j * incColC] +=
                        alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}

void
dgemm_jil(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgescal(m, n, beta, C, incRowC, incColC);
    if (m == 0 || n == 0 || k == 0 || alpha == 0) {
        return;
    }

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t l = 0; l < k; ++l) {
                C[i * incRowC + j * incColC] +=
                        alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}

//------------------------------------------------------------------------------

#define MAX(x, y)    ((x)>(y)) ? (x) : (y)

double
dgemm_err_est(size_t m, size_t n, size_t k,
              double alpha,
              const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
              const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
              const double *C0, ptrdiff_t incRowC0, ptrdiff_t incColC0,
              double beta,
              const double *C_, ptrdiff_t incRowC_, ptrdiff_t incColC_,
              double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    dgeaxpy(m, n, -1, C_, incRowC_, incColC_, C, incRowC, incColC);

    double normC = dgenorm_inf(m, n, C, incRowC, incColC);
    double normC0 = dgenorm_inf(m, n, C0, incRowC0, incColC0);
    double normA = dgenorm_inf(m, k, A, incRowA, incColA);
    double normB = dgenorm_inf(k, n, B, incRowB, incColB);
    size_t N = MAX(m, MAX(n, k));

    return normC / (DBL_EPSILON * (N * fabs(alpha) * normA * normB + fabs(beta) * normC0));
}

void
dgemm_ref(size_t m, size_t n, size_t k,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC) {
    if (beta != 1) {
        if (beta != 0) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < m; ++i) {
                    C[i * incRowC + j * incColC] *= beta;
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < m; ++i) {
                    C[i * incRowC + j * incColC] = 0;
                }
            }
        }
    }
    if (k == 0 || alpha == 0) {
        return;
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            for (size_t i = 0; i < m; ++i) {
                C[i * incRowC + j * incColC] += alpha * A[i * incRowA + l * incColA]
                                                * B[l * incRowB + j * incColB];
            }
        }
    }
}

//------------------------------------------------------------------------------

double
wallTime() {
    static int ticks_per_second = 0;
    if (!ticks_per_second) {
        ticks_per_second = sysconf(_SC_CLK_TCK);
    }
    struct tms timebuf;
    /* times returns the number of real time ticks passed since start */
    return (double) times(&timebuf) / ticks_per_second;
}

//------------------------------------------------------------------------------

#ifndef DIM_M
#define DIM_M 500
#endif

#ifndef DIM_N
#define DIM_N 600
#endif

#ifndef DIM_K
#define DIM_K 700
#endif

#ifndef ALPHA
#define ALPHA 1.5
#endif

#ifndef BETA
#define BETA 2.0
#endif

double A[DIM_M * DIM_K];
double B[DIM_K * DIM_N];
double C0[DIM_M * DIM_N];
double C_[DIM_M * DIM_N];
double C[DIM_M * DIM_N];

const bool rowMajorA[] = {0, 1, 0, 1, 0, 1, 0, 1};
const bool rowMajorB[] = {0, 0, 1, 1, 0, 0, 1, 1};
const bool rowMajorC[] = {0, 0, 0, 0, 1, 1, 1, 1};
const size_t numTests = sizeof(rowMajorA) / sizeof(bool);


int
main() {
    printf("A is %dx%d\n", DIM_M, DIM_K);
    printf("B is %dx%d\n", DIM_K, DIM_N);
    printf("C is %dx%d\n", DIM_M, DIM_N);

    // start header line:
    printf("%4s %4s %4s ", "A", "B", "C");
    // add test cases:
    printf("%14s %14s ", "gemm_ijl", "time");
    printf("%14s %14s ", "gemm_ilj", "time");
    printf("%14s %14s ", "gemm_lij", "time");
    printf("%14s %14s ", "gemm_jil", "time");
    printf("%14s %14s ", "gemm_jli", "time");
    printf("%14s %14s ", "gemm_lji", "time");
    // end header line:
    printf("\n");

    for (int test = 0; test < numTests; ++test) {

        ptrdiff_t incRowA = rowMajorA[test] ? DIM_K : 1;
        ptrdiff_t incColA = rowMajorA[test] ? 1 : DIM_M;

        ptrdiff_t incRowB = rowMajorB[test] ? DIM_N : 1;
        ptrdiff_t incColB = rowMajorB[test] ? 1 : DIM_K;

        ptrdiff_t incRowC = rowMajorC[test] ? DIM_N : 1;
        ptrdiff_t incColC = rowMajorC[test] ? 1 : DIM_M;

        printf("%4s ", incRowA > incColA ? "RM" : "CM");
        printf("%4s ", incRowB > incColB ? "RM" : "CM");
        printf("%4s ", incRowC > incColC ? "RM" : "CM");

        srand(0);
        initMatrix(DIM_M, DIM_K, A, incRowA, incColA, ALPHA == 0.0);
        initMatrix(DIM_K, DIM_N, B, incRowB, incColB, ALPHA == 0.0);
        initMatrix(DIM_M, DIM_N, C0, incRowC, incColC, BETA == 0.0);

        // compute reference solution
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C_, incRowC, incColC);
        dgemm_ref(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C_, incRowC, incColC);

        // test other gemm implementations

        double est, t;

        // dgemm_ijl
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C, incRowC, incColC);

        t = wallTime();
        dgemm_ijl(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C, incRowC, incColC);
        t = wallTime() - t;

        est = dgemm_err_est(DIM_M, DIM_N, DIM_K,
                            ALPHA,
                            A, incRowA, incColA,
                            B, incRowB, incColB,
                            C0, incRowC, incColC,
                            BETA,
                            C_, incRowC, incColC,
                            C, incRowC, incColC);

        printf("%14.2e %14.2lf ", est, t);

        // dgemm_ilj
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C, incRowC, incColC);

        t = wallTime();
        dgemm_ilj(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C, incRowC, incColC);
        t = wallTime() - t;

        est = dgemm_err_est(DIM_M, DIM_N, DIM_K,
                            ALPHA,
                            A, incRowA, incColA,
                            B, incRowB, incColB,
                            C0, incRowC, incColC,
                            BETA,
                            C_, incRowC, incColC,
                            C, incRowC, incColC);

        printf("%14.2e %14.2lf ", est, t);

        // dgemm_lij
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C, incRowC, incColC);

        t = wallTime();
        dgemm_lij(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C, incRowC, incColC);
        t = wallTime() - t;

        est = dgemm_err_est(DIM_M, DIM_N, DIM_K,
                            ALPHA,
                            A, incRowA, incColA,
                            B, incRowB, incColB,
                            C0, incRowC, incColC,
                            BETA,
                            C_, incRowC, incColC,
                            C, incRowC, incColC);

        printf("%14.2e %14.2lf ", est, t);

        // dgemm_jil
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C, incRowC, incColC);

        t = wallTime();
        dgemm_jil(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C, incRowC, incColC);
        t = wallTime() - t;

        est = dgemm_err_est(DIM_M, DIM_N, DIM_K,
                            ALPHA,
                            A, incRowA, incColA,
                            B, incRowB, incColB,
                            C0, incRowC, incColC,
                            BETA,
                            C_, incRowC, incColC,
                            C, incRowC, incColC);

        printf("%14.2e %14.2lf ", est, t);

        // dgemm_jli
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C, incRowC, incColC);

        t = wallTime();
        dgemm_jli(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C, incRowC, incColC);
        t = wallTime() - t;

        est = dgemm_err_est(DIM_M, DIM_N, DIM_K,
                            ALPHA,
                            A, incRowA, incColA,
                            B, incRowB, incColB,
                            C0, incRowC, incColC,
                            BETA,
                            C_, incRowC, incColC,
                            C, incRowC, incColC);

        printf("%14.2e %14.2lf ", est, t);

        // dgemm_lji
        dgecopy(DIM_M, DIM_N, C0, incRowC, incColC, C, incRowC, incColC);

        t = wallTime();
        dgemm_jli(DIM_M, DIM_N, DIM_K,
                  ALPHA,
                  A, incRowA, incColA,
                  B, incRowB, incColB,
                  BETA,
                  C, incRowC, incColC);
        t = wallTime() - t;

        est = dgemm_err_est(DIM_M, DIM_N, DIM_K,
                            ALPHA,
                            A, incRowA, incColA,
                            B, incRowB, incColB,
                            C0, incRowC, incColC,
                            BETA,
                            C_, incRowC, incColC,
                            C, incRowC, incColC);

        printf("%14.2e %14.2lf ", est, t);


        printf("\n");
    }
}
