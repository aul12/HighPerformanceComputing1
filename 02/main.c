#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "matrix.h"

#define ROW_MAJOR false

#define INC_ROW(rowMajor, n) ((rowMajor) ? (n) : 1)
#define INC_COL(rowMajor, m) ((rowMajor) ? (1) : m)

int main() {
    size_t m = 7, n = 8;
    ptrdiff_t incRow = INC_ROW(ROW_MAJOR, n);
    ptrdiff_t incCol = INC_COL(ROW_MAJOR, m);

    double *A = malloc(m * n * sizeof(double));
    if (NULL == A) {
        return -1;
    }

    initMatrix(A, m, n, incRow, incCol);
    printMatrix(A, m, n, incRow, incCol);

    printf("\n\nTransposed:\n");
    printMatrix(A, n, m, INC_ROW(!ROW_MAJOR, m), INC_COL(!ROW_MAJOR, n));

    printf("\n\nMemory:\n");
    printMatrix(A, n*m, 1, 1, 0);

    printf("\n\nSecond Row:\n");
    printMatrix(A + INC_ROW(ROW_MAJOR, n), 1, n, INC_ROW(ROW_MAJOR, n), INC_COL(ROW_MAJOR, m));

    printf("\n\nThird Column:\n");
    printMatrix(A + INC_COL(ROW_MAJOR, m) * 2, m, 1, INC_ROW(ROW_MAJOR, n), INC_COL(ROW_MAJOR, m));

    printf("\n\n2x3 Block (from 2,4):\n");
    printMatrix(A + INC_ROW(ROW_MAJOR, n) * 2 + INC_COL(ROW_MAJOR, m) * 4,
                2, 3, INC_ROW(ROW_MAJOR, n), INC_COL(ROW_MAJOR, m));

    printf("\n\n2x3 Block (from 3,6):\n");
    printMatrix(A + INC_ROW(ROW_MAJOR, n) * 3 + INC_COL(ROW_MAJOR, m) * 6,
                2, 3, INC_ROW(ROW_MAJOR, n), INC_COL(ROW_MAJOR, m));

    free(A);

    return 0;
}
