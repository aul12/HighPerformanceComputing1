#include <cassert>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <cfloat>
#include <printf.hpp>
#include <sys/times.h>
#include <unistd.h>

//------------------------------------------------------------------------------

#include "gemm.hpp"

//------------------------------------------------------------------------------

namespace test{

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


    void
    randMatrix(std::size_t m, std::size_t n,
               double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] = ((double)rand() - RAND_MAX/2)*2/RAND_MAX;
            }
        }
    }

    void
    nanMatrix(std::size_t m, std::size_t n,
              double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] = std::nan("");
            }
        }
    }

    void
    printMatrix(std::size_t m, std::size_t n,
                const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        for (std::size_t i=0; i<m; ++i) {
            for (std::size_t j=0; j<n; ++j) {
                fmt::printf("%10.3lf ", A[i*incRowA+j*incColA]);
            }
            fmt::printf("\n");
        }
        fmt::printf("\n");
    }

    double
    genorm_inf(std::size_t m, std::size_t n,
               const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        double res = 0;
        for (std::size_t i=0; i<m; ++i) {
            double asum = 0;
            for (std::size_t j=0; j<n; ++j) {
                asum += std::fabs(A[i*incRowA+j*incColA]);
            }
            if (std::isnan(asum)) {
                return asum;
            }
            if (asum>res) {
                res = asum;
            }
        }
        return res;
    }

    void
    gescal(std::size_t m, std::size_t n, double alpha,
           double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
    {
        if (m==0 || n==0 || alpha==1) {
            return;
        }
        // A is row major: scale A^T
        if (incRowA>incColA) {
            gescal(n, m, alpha, A, incColA, incRowA);
            return;
        }
        // A is col major:
        if (alpha!=0) {
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    A[i*incRowA+j*incColA] *= alpha;
                }
            }
        } else {
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    A[i*incRowA+j*incColA] = 0;
                }
            }
        }
    }

    void
    geaxpy(std::size_t m, std::size_t n, double alpha,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
    {
        if (m==0 || n==0 || alpha==0) {
            return;
        }
        // if B is row major:   B^T <- alpha*A^T + B^T
        if (incRowB>incColB) {
            geaxpy(n, m, alpha, A, incColA, incRowA, B, incColB, incRowB);
            return;
        }
        // B is col major:
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] += alpha*A[i*incRowA+j*incColA];
            }
        }
    }

    void
    gecopy(std::size_t m, std::size_t n,
           const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
           double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
    {
        if (m==0 || n==0) {
            return;
        }
        // if B is row major:   B^T <- A^T
        if (incRowB>incColB) {
            gecopy(n, m, A, incColA, incRowA, B, incColB, incRowB);
            return;
        }
        // B is col major:
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                B[i*incRowB+j*incColB] = A[i*incRowA+j*incColA];
            }
        }
    }

#define MAX(x,y)    ((x)>(y)) ? (x) : (y)

    double
    gemm_err_est(std::size_t m, std::size_t n, std::size_t k,
                 double alpha,
                 const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
                 const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
                 const double *C0, std::ptrdiff_t incRowC0, std::ptrdiff_t incColC0,
                 double beta,
                 const double *C_, std::ptrdiff_t incRowC_, std::ptrdiff_t incColC_,
                 double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
    {
        geaxpy(m, n, -1, C_, incRowC_, incColC_, C, incRowC, incColC);

        double normD  = genorm_inf(m, n, C, incRowC, incColC);
        std::size_t N = MAX(m, MAX(n, k));

        if (std::isnan(normD)) {
            return normD;
        }

        if (normD==0) {
            return 0;
        }

        double normA = 0;
        double normB = 0;

        if (alpha!=0) {
            normB  = genorm_inf(k, n, B, incRowB, incColB);
            normA  = genorm_inf(m, k, A, incRowA, incColA);
            normA  *= fabs(alpha);
        }

        double normC0 = 0;
        if (beta!=0) {
            normC0 = genorm_inf(m, n, C0, incRowC0, incColC0);
            normC0 *= fabs(beta);
        }

        return normD/(DBL_EPSILON*(N*normA*normB+normC0));
    }

} // namespace test

//------------------------------------------------------------------------------

void
gemm_ref(std::size_t m, std::size_t n, std::size_t k,
         double alpha,
         const double *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
         const double *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
         double beta,
         double *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
{
    if (beta!=1) {
        if (beta!=0) {
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    C[i*incRowC+j*incColC] *= beta;
                }
            }
        } else {
            for (std::size_t j=0; j<n; ++j) {
                for (std::size_t i=0; i<m; ++i) {
                    C[i*incRowC+j*incColC] = 0;
                }
            }
        }
    }
    if (k==0 || alpha==0) {
        return;
    }
    for (std::size_t j=0; j<n; ++j) {
        for (std::size_t l=0; l<k; ++l) {
            for (std::size_t i=0; i<m; ++i) {
                C[i*incRowC+j*incColC] += alpha*A[i*incRowA+l*incColA]
                                          *B[l*incRowB+j*incColB];
            }
        }
    }
}

//------------------------------------------------------------------------------

#ifndef ALPHA
#define ALPHA   1
#endif

#ifndef BETA
#define BETA    1
#endif

#ifndef DIM_MAX_M
#define DIM_MAX_M   1000
#endif

#ifndef DIM_MAX_N
#define DIM_MAX_N   1000
#endif

#ifndef DIM_MAX_K
#define DIM_MAX_K   1000
#endif

#ifndef COLMAJOR_C
#define COLMAJOR_C 1
#endif

#ifndef COLMAJOR_A
#define COLMAJOR_A 1
#endif

#ifndef COLMAJOR_B
#define COLMAJOR_B 1
#endif

int
main()
{
    fmt::printf("#Configuration:\n");
    fmt::printf("#\tMC = %5zu\n", ulmblas::dgemm_parameter::MC);
    fmt::printf("#\tNC = %5zu\n", ulmblas::dgemm_parameter::NC);
    fmt::printf("#\tKC = %5zu\n", ulmblas::dgemm_parameter::KC);
    fmt::printf("#\tMR = %5zu\n", ulmblas::dugemm_parameter::MR);
    fmt::printf("#\tNR = %5zu\n", ulmblas::dugemm_parameter::NR);


    fmt::printf("#\n");
    fmt::printf("#Benchmark:\n");
    fmt::printf("#%7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %12s "
                "%7s %7s %12s %12s\n",
                "MR", "NR", "k", "incRowC", "incColC",
                "incRowA", "incColA", "incRowB", "incColB",
                "alpha", "beta", "error", "tRef", "tTst",
                "mflops: ref", "tst");
    for (std::size_t m=100, n=100, k=100;
         m <= DIM_MAX_M && n <= DIM_MAX_N && k <= DIM_MAX_K;
         m += 100, n +=100, k += 100)
    {
        std::size_t incRowC = (COLMAJOR_C) ? 1 : n;
        std::size_t incColC = (COLMAJOR_C) ? m : 1;

        std::size_t incRowA = (COLMAJOR_A) ? 1 : k;
        std::size_t incColA = (COLMAJOR_A) ? m : 1;

        std::size_t incRowB = (COLMAJOR_B) ? 1 : n;
        std::size_t incColB = (COLMAJOR_B) ? k : 1;

        tools::DoubleArray A(m*k);
        tools::DoubleArray B(k*n);
        tools::DoubleArray C0(m*n);
        tools::DoubleArray Cref(m*n);
        tools::DoubleArray Ctst(m*n);

        if (BETA==0) {
            test::nanMatrix(m, n, C0, incRowC, incColC);
        } else {
            test::randMatrix(m, n, C0, incRowC, incColC);
        }
        if (ALPHA==0) {
            test::nanMatrix(m, k, A, incRowA, incColA);
            test::nanMatrix(k, n, B, incRowB, incColB);
        } else {
            test::randMatrix(m, k, A, incRowA, incColA);
            test::randMatrix(k, n, B, incRowB, incColB);
        }

        // call reference implementation
        test::gecopy(m, n,
                     C0, incRowC, incColC,
                     Cref, incRowC, incColC);
        double tRef = test::wallTime();
        gemm_ref(m, n, k,
                 ALPHA,
                 A, incRowA, incColA,
                 B, incRowB, incColB,
                 BETA,
                 Cref, incRowC, incColC);
        tRef = test::wallTime() - tRef;

        // call, test and bench other implementation
        test::gecopy(m, n,
                     C0, incRowC, incColC,
                     Ctst, incRowC, incColC);

        double tTst = test::wallTime();
        ulmblas::gemm(m, n, k,
                      ALPHA,
                      A, incRowA, incColA,
                      B, incRowB, incColB,
                      BETA,
                      Ctst, incRowC, incColC);
        tTst = test::wallTime() - tTst;

        double err = test::gemm_err_est(m, n, k,
                                        ALPHA,
                                        A, incRowA, incColA,
                                        B, incRowB, incColB,
                                        C0, incRowC, incColC,
                                        BETA,
                                        Cref, incRowC, incColC,
                                        Ctst, incRowC, incColC);

        double mflop = 2.*m/1000*n/1000*k;

        fmt::printf(" %7zu %7zu %7zu %7zu %7zu %7zu %7zu %7zu %7zu "
                    "%7.2lf %7.2lf %12.2e %7.2lf %7.2lf %12.2lf %12.2lf\n",
                    m, n, k, incRowC, incColC, incRowA, incColA,
                    incRowB, incColB, ALPHA, BETA, err, tRef, tTst,
                    mflop/tRef, mflop/tTst);
    }
}
