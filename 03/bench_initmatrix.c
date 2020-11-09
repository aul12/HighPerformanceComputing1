#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <math.h>


#ifndef MINDIM_M
#define MINDIM_M    1000
#endif

#ifndef MINDIM_N
#define MINDIM_N    1000
#endif

#ifndef MINDIM_K
#define MINDIM_K    1000
#endif

#ifndef MAXDIM_M
#define MAXDIM_M    7000
#endif

#ifndef MAXDIM_N
#define MAXDIM_N    7000
#endif

#ifndef MAXDIM_K
#define MAXDIM_K    7000
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

int
main()
{
    double *buffer1 = malloc(MAXDIM_M*MAXDIM_N*sizeof(double));
    double *buffer2 = malloc(MAXDIM_M*MAXDIM_N*sizeof(double));

    if (!buffer1 || !buffer2) {
        printf("allocating memory failed!\n");
        free(buffer1);
        free(buffer2);
        return 1;
    }

    printf("   M    N      t1      t2   t2/t1       diff\n");
    printf("          col-maj row-maj\n");
    printf("============================================\n");

    for (size_t m = MINDIM_M, n = MINDIM_N;
         m < MAXDIM_M && n < MAXDIM_N;
         m += INC_M, n += INC_N)
    {
        size_t runs = 0;
        double t1 = 0;
        do {
            double t0 = wallTime();
            initMatrix(m, n, buffer1, 1, m);
            t1 += wallTime() - t0;
            ++runs;
        } while (t1 < MIN_T);
        t1 /= runs;

        runs = 0;
        double t2 = 0;
        do {
            double t0 = wallTime();
            initMatrix(m, n, buffer2, n, 1);
            t2 += wallTime() - t0;
            ++runs;
        } while (t2 < MIN_T);
        t2 /= runs;

        double diff = asumDiff(m, n, buffer1, 1, m, buffer2, n, 1);

        printf("%4zd %4zd %7.2lf %7.2lf %7.2lf %10.2le\n",
               m, n, t1, t2, t2/t1, diff);
    }
    free(buffer1);
    free(buffer2);
}
