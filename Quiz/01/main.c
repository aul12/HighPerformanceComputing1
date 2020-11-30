#include <stdio.h>
#include <memory.h>

#include "quiz01.h"

#define M 3
#define N 4

void test(const double *A, const double *x, const double *y, double alpha, double beta, const double *y_true) {
    double y_tmp[M];
    memcpy(y_tmp, y, M * sizeof(double));

    dgemv(M, N, alpha, A, N, 1, x, 1, beta, y_tmp, 1);

    if (memcmp(y_tmp, y_true, M * sizeof(double)) != 0) {
        printf("Failure!\n");
    }
}

int main() {
    double A[N*M] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
    };

    double x[N] = {1, 2, 3, 4};
    double y[M] = {7, 8, 9};

    test(A, x, y, 0, 1, y);
    test(A, x, y, 0, 2, (double[]){14, 16, 18});
    test(A, x, y, 0, 0, (double[]){0, 0, 0});

    test(A, x, y, 1, 0, (double[]){
        1*1+2*2+3*3+4*4,
        1*5+2*6+3*7+4*8,
        1*9+2*10+3*11+4*12,
    });

    test(A, x, y, 2, 0, (double[]){
            2*(1*1+2*2+3*3+4*4),
            2*(1*5+2*6+3*7+4*8),
            2*(1*9+2*10+3*11+4*12),
    });

    test(A, x, y, 1, 1, (double[]){
            1*1+2*2+3*3+4*4+7,
            1*5+2*6+3*7+4*8+8,
            1*9+2*10+3*11+4*12+9,
    });

    return 0;
}
