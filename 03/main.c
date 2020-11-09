#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void
initMatrix(size_t m, size_t n, double *A, ptrdiff_t incRowA, ptrdiff_t incColA)
{
    for (size_t j=0; j<n; ++j) {
        for (size_t i=0; i<m; ++i) {
            A[i*incRowA+j*incColA] = i*n + j + 1;
        }
    }
}

void
printMatrix(size_t m, size_t n,
            const double *A, ptrdiff_t incRowA, ptrdiff_t incColA)
{
    for (size_t i=0; i<m; ++i) {
        for (size_t j=0; j<n; ++j) {
            printf("%8.3lf ", A[i*incRowA+j*incColA]);
        }
        printf("\n");
    }
    printf("\n");
}


// function dasum
double dasum(size_t n, const double *x, ptrdiff_t inc) {
    double res = 0;
    for (size_t c=0; c<n; ++c) {
        res += x[c * inc];
    }
    return res;
}


// function dswap
void dswap(size_t n, double *x, double *y, ptrdiff_t incX, ptrdiff_t incY) {
    for (size_t c=0; c<n; ++c) {
        double tmp = y[c * incY];
        y[c * incY] = x[c*incX];
        x[c*incX] = tmp;
    }
}

// function daxpy
void daxpy(size_t n, double *y, const double *x, double a, ptrdiff_t incX, ptrdiff_t incY) {
    for (size_t c=0; c<n; ++c) {
        y[c * incY] += a * x[c * incX];
    }
}

int
main()
{
    size_t m = 7;
    size_t n = 8;

    ptrdiff_t incRowA = 1;
    ptrdiff_t incColA = m;

    double *A = malloc(m*n*sizeof(double));

    initMatrix(m, n, A, incRowA, incColA);

    printf("A = \n");
    printMatrix(m, n, A, incRowA, incColA);

    // Exercise 3.1:
    // -------------

    printf("asum(A(1,:)) = %lf\n", dasum(n, A+incRowA, incColA));
    printf("asum(A(:,2)) = %lf\n", dasum(m, A+2*incColA, incRowA));

    size_t mn = (m<n) ? m : n;  // length of diagonal
    printf("asum(diag(A)) = %lf\n", dasum(mn, A, incRowA + incColA));

    // Exercise 3.2:
    // -------------
    dswap(n, A+incRowA*1, A+incRowA*2, incColA, incColA);
    printf("After dswap(A(1,:), A(2,:))\n");
    printMatrix(m, n, A, incRowA, incColA);

    dswap(m, A+incColA*1, A+incColA*2, incRowA, incRowA);
    printf("After dswap(A(:,1), A(:,2))\n");
    printMatrix(m, n, A, incRowA, incColA);

    // Exercise 3.3:
    // -------------
    daxpy(m, A+incColA*2, A+incColA*1, 3, incRowA, incRowA);
    printf("After daxpy(3.0, A(:,1), A(:,2))\n");
    printMatrix(m, n, A, incRowA, incColA);

    daxpy(n, A+incRowA*3, A+incRowA*1, -2.5, incColA, incColA);
    printf("After daxpy(-2.5, A(1,:), A(3,:))\n");
    printMatrix(m, n, A, incRowA, incColA);

    free(A);

    return 0;
}
