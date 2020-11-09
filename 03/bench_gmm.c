#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <math.h>
#include <stddef.h>


#ifndef MINDIM_M
#define MINDIM_M    100
#endif

#ifndef MINDIM_N
#define MINDIM_N    100
#endif

#ifndef MINDIM_K
#define MINDIM_K    100
#endif

#ifndef MAXDIM_M
#define MAXDIM_M    1000
#endif

#ifndef MAXDIM_N
#define MAXDIM_N    1000
#endif

#ifndef MAXDIM_K
#define MAXDIM_K    1000
#endif

#ifndef INC_M
#define INC_M   100
#endif

#ifndef INC_N
#define INC_N   100
#endif

#ifndef INC_K
#define INC_K   100
#endif

#ifndef MIN_T
#define MIN_T   1
#endif

#ifndef ALPHA
#define ALPHA   1
#endif

#ifndef BETA
#define BETA   1
#endif


/* return real time in seconds since start of the process */
double
wallTime()
{
    static int ticks_per_second = 0;
    if (!ticks_per_second) {
        ticks_per_second = sysconf(_SC_CLK_TCK);
    }
    struct tms timebuf;
    /* times returns the number of real time ticks passed since start */
    return (double) times(&timebuf) / ticks_per_second;
}

double
asumDiff(size_t m, size_t n, const double *A, size_t incRowA, size_t incColA,
         const double *B, size_t incRowB, size_t incColB)
{
    double diff = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            diff += fabs(B[i*incRowB+j*incColB] - A[i*incRowA+j*incColA]);
        }
    }
    return diff;
}

void
initMatrix(size_t m, size_t n, double *A, size_t incRowA, size_t incColA)
{
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A[i*incRowA+j*incColA] = j*n+i+1;
        }
    }
}

void
printMatrix(size_t m, size_t n, const double *A,
            size_t incRowA, size_t incColA)
{
    for (size_t i = 0; i < m; ++i) {
        printf("   ");
        for (size_t j = 0; j < n; ++j) {
            printf("%4.1lf ", A[i*incRowA+j*incColA]);
        }
        printf("\n");
    }
    printf("\n");
}

void
dgemm_row(size_t m, size_t n, size_t k, double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC)
{
    if (beta!=0) {
        if (beta!=1) {
            for (size_t i=0; i<m; ++i) {
                for (size_t j=0; j<n; ++j) {
                    C[i*incRowC+j*incColC] *= beta;
                }
            }
        }
    } else {
        for (size_t i=0; i<m; ++i) {
            for (size_t j=0; j<n; ++j) {
                C[i*incRowC+j*incColC] = 0;
            }
        }
    }
    if (alpha==0 || k==0) {
        return;
    }
    for (size_t i=0; i<m; ++i) {
        for (size_t l=0; l<k; ++l) {
            for (size_t j=0; j<n; ++j) {
                C[i*incRowC+j*incColC] +=
                        alpha*A[i*incRowA+l*incColA]*B[l*incRowB+j*incColB];
            }
        }
    }
}

void
dgemm_col(size_t m, size_t n, size_t k, double alpha,
          const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
          const double *B, ptrdiff_t incRowB, ptrdiff_t incColB,
          double beta,
          double *C, ptrdiff_t incRowC, ptrdiff_t incColC)
{
    for (size_t i=0; i<m; ++i) {
        for (size_t j=0; j<n; ++j) {
            C[i * incRowC + j * incColC] *= beta;
        }
    }

    for (size_t j=0; j<n; ++j) {
        for (size_t l=0; l<k; ++l) {
            for (size_t i=0; i<m; ++i) {
                C[i * incRowC + j * incColC] += alpha * A[i * incRowA + l * incColA] * B[l * incRowB + j * incColB];
            }
        }
    }
}


int
main()
{
    double *A = malloc(MAXDIM_M*MAXDIM_K*sizeof(double));
    double *B = malloc(MAXDIM_K*MAXDIM_N*sizeof(double));
    double *C1 = malloc(MAXDIM_M*MAXDIM_N*sizeof(double));
    double *C2 = malloc(MAXDIM_M*MAXDIM_N*sizeof(double));

    if (!A || !B || !C1 || !C2) {
        printf("memory allocation failed!\n");
        free(A);
        free(B);
        free(C1);
        free(C2);
        return 1;
    }

    printf("   M    N      t1      t2   t2/t1       diff\n");
    printf("          col-maj row-maj\n");
    printf("============================================\n");

    for (size_t m = MINDIM_M, n = MINDIM_N, k = MINDIM_K;
         m <= MAXDIM_M && n <= MAXDIM_N && k <= MAXDIM_K;
         m += INC_M, n += INC_N, k += INC_K)
    {
        // set storage order for A, B, C1, C2
        ptrdiff_t incRowA = 1, incColA = m;
        ptrdiff_t incRowB = 1, incColB = k;
        ptrdiff_t incRowC = 1, incColC = m;     // used for C1, C2

        initMatrix(m, k, A, incRowA, incColA);
        initMatrix(k, n, B, incRowB, incColB);

        size_t runs = 0;
        double t1 = 0;
        do {
            initMatrix(m, n, C1, incRowC, incColC);

            double t0 = wallTime();
            dgemm_col(m, n, k, ALPHA,
                      A, incRowA, incColA,
                      B, incRowB, incColB,
                      BETA,
                      C1, incRowC, incColC);
            t1 += wallTime() - t0;
            ++runs;
        } while (t1 < MIN_T);
        t1 /= runs;

        runs = 0;
        double t2 = 0;
        do {
            initMatrix(m, n, C2, incRowC, incColC);

            double t0 = wallTime();
            dgemm_row(m, n, k,
                      ALPHA,
                      A, incRowA, incColA,
                      B, incRowB, incColB,
                      BETA,
                      C2, incRowC, incColC);
            t2 += wallTime() - t0;
            ++runs;
        } while (t2 < MIN_T);
        t2 /= runs;

        double diff = asumDiff(m, n,
                               C1, incRowC, incColC,
                               C2, incRowC, incColC);

        printf("%4zd %4zd %zd %7.2lf %7.2lf %7.2lf %10.2le\n",
               m, n, k, t1, t2, t2/t1, diff);
    }

    free(A);
    free(B);
    free(C1);
    free(C2);
}
