#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <stdbool.h>

//------------------------------------------------------------------------------

#define MY_ABS(x)   ((x)<0 ? -(x) : (x))

//------------------------------------------------------------------------------

void
initMatrix(size_t m, size_t n, double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
           bool withNan)
{
    // if A is row major initialize A^T
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
        initMatrix(n, m, A, incColA, incRowA, withNan);
        return;
    }
    // if A is col major
    if (withNan) {
        for (size_t j=0; j<n; ++j) {
            for (size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] = nan("");
            }
        }
    } else {
        for (size_t j=0; j<n; ++j) {
            for (size_t i=0; i<m; ++i) {
                double rValue = ((double)rand() - RAND_MAX/2)*2/RAND_MAX;
                A[i*incRowA+j*incColA] = rValue;
            }
        }
    }
}

void
printMatrix(size_t m, size_t n,
            const double *A, ptrdiff_t incRowA, ptrdiff_t incColA)
{
    for (size_t i=0; i<m; ++i) {
        for (size_t j=0; j<n; ++j) {
            printf("%10.3lf ", A[i*incRowA+j*incColA]);
        }
        printf("\n");
    }
    printf("\n");
}

//------------------------------------------------------------------------------

void
dgeaxpy(size_t m, size_t n, double alpha,
        const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
        double *B, ptrdiff_t incRowB, ptrdiff_t incColB)
{
    if (alpha == 0) {
        return;
    }

    // Row major
    if (MY_ABS(incRowB) > MY_ABS(incColB)) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                B[i * incRowA + j * incColA] += alpha * A[i * incRowA + j * incColA];
            }
        }
    } else {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                B[i * incRowA + j * incColA] += alpha * A[i * incRowA + j * incColA];
            }
        }
    }
}

void
dgecopy(size_t m, size_t n,
        const double *A, ptrdiff_t incRowA, ptrdiff_t incColA,
        double *B, ptrdiff_t incRowB, ptrdiff_t incColB)
{
    // Row major
    if (MY_ABS(incRowB) > MY_ABS(incColB)) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                B[i * incRowA + j * incColA] = A[i * incRowA + j * incColA];
            }
        }
    } else {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                B[i * incRowA + j * incColA] = A[i * incRowA + j * incColA];
            }
        }
    }
}

void
dgescal(size_t m, size_t n, double alpha,
        double *A, ptrdiff_t incRowA, ptrdiff_t incColA)
{
    if (alpha == 1) {
        return;
    }

    // Row major
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                A[i * incRowA + j * incColA] *= alpha;
            }
        }
    } else {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                A[i * incRowA + j * incColA] *= alpha;
            }
        }
    }
}

double
dgenorm_inf(size_t m, size_t n,
            const double *A, ptrdiff_t incRowA, ptrdiff_t incColA)
{
    double max = -INFINITY;

    // Row major
    if (MY_ABS(incRowA) > MY_ABS(incColA)) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (max < A[i * incRowA + j * incColA]) {
                    max = A[i * incRowA + j * incColA];
                }
            }
        }
    } else {
        for (int j=0; j<n; ++j) {
            for (int i=0; i<m; ++i) {
                if (max < A[i * incRowA + j * incColA]) {
                    max = A[i * incRowA + j * incColA];
                }
            }
        }
    }

    return max;
}


//------------------------------------------------------------------------------

#ifndef DIM_M
#define DIM_M 4
#endif

#ifndef DIM_N
#define DIM_N 5
#endif

double A[DIM_M*DIM_N];
double B[DIM_M*DIM_N];

const bool rowMajorA[] = {false, true};
const bool rowMajorB[] = {false, false};
const size_t numTests = sizeof(rowMajorA)/sizeof(bool);

int
main()
{
    for (size_t test=0; test<numTests; ++test) {
        ptrdiff_t incRowA = rowMajorA[test] ? DIM_N : 1;
        ptrdiff_t incColA = rowMajorA[test] ? 1 : DIM_M;

        ptrdiff_t incRowB = rowMajorB[test] ? DIM_N : 1;
        ptrdiff_t incColB = rowMajorB[test] ? 1 : DIM_M;

        printf("Matrix A is stored %s\n", rowMajorA[test] ? "row major"
                                                          : "col major");
        printf("Matrix B is stored %s\n", rowMajorB[test] ? "row major"
                                                          : "col major");
        srand(0);
        initMatrix(DIM_M, DIM_N, A, incRowA, incColA, false);
        initMatrix(DIM_M, DIM_N, B, incRowB, incColB, false);

        printf("A =\n");
        printMatrix(DIM_M, DIM_N, A, incRowA, incColA);
        printf("B =\n");
        printMatrix(DIM_M, DIM_N, B, incRowB, incColB);

        printf("B = 2*B\n");
        // Call dgescal for: B <- 2*B
        dgescal(DIM_M, DIM_N, 2, B, incRowB, incColB);
        printf("B =\n");
        printMatrix(DIM_M, DIM_N, B, incRowB, incColB);

        printf("||A||_inf = %lf\n", dgenorm_inf(DIM_M, DIM_N, A, incRowA, incColA));
        printf("||B||_inf = %lf\n", dgenorm_inf(DIM_M, DIM_N, B, incRowB, incColB));

        // Call dgeaxpy for: B <- B - A
        dgeaxpy(DIM_M, DIM_N, -1, A, incRowA, incColA, B, incRowB, incColB);
        printf("||B-A||_inf = %lf\n",
               dgenorm_inf(DIM_M, DIM_N, B, incRowB, incColB));

        // Call dgecopy for: B <- A
        dgecopy(DIM_M, DIM_N, A, incRowA, incColA, B, incRowB, incColB);
        printf("B = A\n");
        printf("B =\n");
        printMatrix(DIM_M, DIM_N, B, incRowB, incColB);

        // Call dgeaxpy for: B <- B - A
        dgeaxpy(DIM_M, DIM_N, -1, A, incRowA, incColA, B, incRowB, incColB);
        printf("||B-A||_inf = %lf\n",
               dgenorm_inf(DIM_M, DIM_N, B, incRowB, incColB));
    }
}
