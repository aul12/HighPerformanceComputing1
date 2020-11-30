#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/times.h>
#include <unistd.h>
#include <mkl/mkl_types.h>

#ifndef MIN_M
#define MIN_M 100
#endif

#ifndef MIN_N
#define MIN_N 100
#endif

#ifndef MAX_M
#define MAX_M 1500
#endif

#ifndef MAX_N
#define MAX_N 1500
#endif

#ifndef INCX
#define INCX 1
#endif

#ifndef INCY
#define INCY 1
#endif

#ifndef ALPHA
#define ALPHA 1.5
#endif

#ifndef BETA
#define BETA 1.5
#endif

#ifndef T_MIN
#define T_MIN 5
#endif


double A[MAX_M * MAX_N];
double X[MAX_N * INCX];
double Y[MAX_M * INCY];
double Y1[MAX_M * INCY];
double Y2[MAX_M * INCY];
double Y3[MAX_M * INCY];
double Y4[MAX_M * INCY];
double Y5[MAX_M * INCY];

double
walltime() {
    struct tms ts;
    static double ClockTick = 0.0;

    if (ClockTick == 0.0) {
        ClockTick = 1.0 / ((double) sysconf(_SC_CLK_TCK));
    }
    return ((double) times(&ts)) * ClockTick;
}

void
initMatrix(size_t m, size_t n, double *A, size_t incRowA, size_t incColA) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A[i * incRowA + j * incColA] = ((double) rand() - RAND_MAX / 2) * 200 / RAND_MAX;
        }
    }
}

void
copyMatrix(size_t m, size_t n,
           const double *A, size_t incRowA, size_t incColA,
           double *B, size_t incRowB, size_t incColB) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            B[i * incRowB + j * incColB] = A[i * incRowA + j * incColA];
        }
    }
}

double
asumDiffMatrix(size_t m, size_t n,
               const double *A, size_t incRowA, size_t incColA,
               double *B, size_t incRowB, size_t incColB) {
    double asum = 0;

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            asum += fabs(B[i * incRowB + j * incColB] - A[i * incRowA + j * incColA]);
        }
    }
    return asum;
}

//------------------------------------------------------------------------------

void
dgemv(const char *trans,
      const MKL_INT *m, const MKL_INT *n, const double *alpha,
      const double *A, const MKL_INT *ldA, const double *x,
      const MKL_INT *incX,
      const double *beta, double *y, const MKL_INT *incY);

void
dgemv_mkl(MKL_INT m, MKL_INT n,
          double alpha,
          const double *A, MKL_INT incRowA, MKL_INT incColA,
          const double *x, MKL_INT incX,
          double beta,
          double *y, MKL_INT incY) {
    MKL_INT ldA = (incRowA == 1) ? incColA : incRowA;
    char trans = (incRowA == 1) ? 'N' : 'T';
    MKL_INT M = (incRowA == 1) ? m : n;
    MKL_INT N = (incRowA == 1) ? n : m;

    dgemv(&trans, &M, &N, &alpha, A, &ldA, x, &incX, &beta, y, &incY);
}

//------------------------------------------------------------------------------

void dscal_ulm(size_t n, double *x, ptrdiff_t xInc, double alpha) {
    if (alpha == 1) {
        return;
    } else if (alpha == 0) {
        for (size_t c = 0; c < n; ++c) {
            x[c * xInc] = 0;
        }
    } else {
        for (size_t c = 0; c < n; ++c) {
            x[c * xInc] *= alpha;
        }
    }
}

double ddot_ulm(size_t n, const double *x, ptrdiff_t xInc, const double *y, ptrdiff_t yInc) {
    double sum = 0;
    for (size_t c = 0; c < n; ++c) {
        sum += x[c * xInc] * y[c * yInc];
    }
    return sum;
}

void daxpy_ulm(size_t n, double *y, ptrdiff_t yInc, const double *x, ptrdiff_t xInc, double alpha) {
    if (alpha == 0) {
        return;
    } else {
        for (size_t c = 0; c < n; ++c) {
            y[c * yInc] += alpha * x[c * xInc];
        }
    }
}

void dgemv_ulm(size_t m, size_t n,
               double alpha,
               const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
               const double *x, ptrdiff_t incX,
               double beta,
               double *y, ptrdiff_t incY) {
    dscal_ulm(m, y, incY, beta);
    if (incRowA > incColA) {
        for (size_t i = 0; i < m; ++i) {
            y[i * incY] += alpha * ddot_ulm(n, A + i * incRowA, incColA, x, incX);
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            daxpy_ulm(m, y, incY, A + incColA * j, incRowA, alpha * x[j * incX]);
        }
    }
}

//------------------------------------------------------------------------------
#ifndef DGEMV_DOTF_FUSE
#define DGEMV_DOTF_FUSE 12
#endif

#ifndef DGEMV_AXPYF_FUSE
#define DGEMV_AXPYF_FUSE 12
#endif

void
ddotf_ulm(size_t n,
          double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *x, ptrdiff_t incX,
          double *y, ptrdiff_t incY) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < DGEMV_DOTF_FUSE; ++i) {
            y[i * incY] += alpha * A[j * incColA + i * incRowA] * x[j * incX];
        }
    }
}

void
daxpyf_ulm(size_t m,
           double alpha,
           const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
           const double *x, ptrdiff_t incX,
           double *y, ptrdiff_t incY) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < DGEMV_AXPYF_FUSE; ++j) {
            y[i * incY] += alpha * A[j * incColA + i * incRowA] * x[j * incX];
        }
    }
}

void dgemvf_ulm(size_t m, size_t n,
                double alpha,
                const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
                const double *x, ptrdiff_t incX,
                double beta,
                double *y, ptrdiff_t incY) {
    dscal_ulm(m, y, incY, beta);

    if (incRowA > incColA) {
        size_t m_b = m / DGEMV_DOTF_FUSE;
        for (size_t i = 0; i < m_b; ++i) {
            ddotf_ulm(n, alpha, A + i * incRowA * DGEMV_DOTF_FUSE, incRowA, incColA, x, incX, y+i*incY*DGEMV_DOTF_FUSE, incY);
        }

        size_t offset = DGEMV_DOTF_FUSE * m_b;
        for (size_t i = offset; i<m; ++i) {
            y[i * incY] += alpha * ddot_ulm(n, A + i * incRowA, incColA, x, incX);
        }
    } else {
        size_t n_b = n / DGEMV_AXPYF_FUSE;
        for (size_t j = 0; j < n_b; ++j) {
            daxpyf_ulm(m, alpha, A + incColA * j * DGEMV_AXPYF_FUSE, incRowA, incColA, x+j*incX*DGEMV_AXPYF_FUSE, incX, y, incY);
        }

        size_t offset = DGEMV_AXPYF_FUSE * n_b;
        for (size_t j=offset; j<n; ++j) {
            daxpy_ulm(m, y, incY, A + incColA * j, incRowA, alpha * x[j * incX]);
        }
    }
}

//------------------------------------------------------------------------------

#ifndef COLMAJOR
#define COLMAJOR 0
#endif

int
main() {
    size_t runs, incRowA, incColA;
    double t0, t1, t2;
    double diff2;
    double alpha = ALPHA;
    double beta = BETA;

    initMatrix(MAX_M, MAX_N, A, 1, MAX_M);
    initMatrix(MAX_N, 1, X, INCX, 1);
    initMatrix(MAX_M, 1, Y, INCY, 1);

    printf("# COLMAJOR    = %d\n", COLMAJOR);
    printf("# T_MIN       = %d\n", T_MIN);
    printf("#RUN    M     N  INCROW  INCCOL");
    printf("    GEMV_MKL    GEMV_ULM");
    printf("    GEMV_MKL    GEMV_ULM");
    printf("       DIFF2");
    printf("\n");
    printf("#                              ");
    printf("    (t in s)    (t in s)");
    printf("    (MFLOPS)    (MFLOPS)");
    printf("           ");
    printf("\n");

    for (size_t i = 0, m = MIN_M, n = MIN_N; m <= MAX_M && n <= MAX_N;
         ++i, m += 100, n += 100) {

        if (COLMAJOR) {
            incRowA = 1;
            incColA = m;
        } else {
            incRowA = n;
            incColA = 1;
        }

        t1 = 0;
        runs = 0;
        do {
            copyMatrix(MAX_M, 1, Y, INCY, 1, Y1, INCY, 1);
            t0 = walltime();
            dgemv_mkl(m, n, alpha,
                      A, incRowA, incColA,
                      X, INCX,
                      beta,
                      Y1, INCY);
            t1 += walltime() - t0;
            ++runs;
        } while (t1 < T_MIN);
        t1 /= runs;

        t2 = 0;
        runs = 0;
        do {
            copyMatrix(MAX_M, 1, Y, INCY, 1, Y2, INCY, 1);
            t0 = walltime();
            dgemvf_ulm(m, n, alpha,
                      A, incRowA, incColA,
                      X, INCX,
                      beta,
                      Y2, INCY);
            t2 += walltime() - t0;
            ++runs;
        } while (t2 < T_MIN);
        t2 /= runs;
        diff2 = asumDiffMatrix(m, 1, Y1, INCY, 1, Y2, INCY, 1);

        printf("%3ld %5ld %5ld %7ld %7ld ", i, m, n, incRowA, incColA);
        printf("%11.4lf %11.4lf ", t1, t2);
        printf("%11.4lf ", 2 * (m / 1000.0) * (n / 1000.0) / t1);
        printf("%11.4lf ", 2 * (m / 1000.0) * (n / 1000.0) / t2);
        printf("%11.4lf ", diff2);
        printf("\n");
    }
}
