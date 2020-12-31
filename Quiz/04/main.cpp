#include <iostream>

#include "mm-mt.hpp"

constexpr auto N = 7;
constexpr auto M = 1;
constexpr auto K = 9;

using Mat = hpc::matvec::GeMatrix<double>;

int main() {
    Mat A{M, K};
    Mat B{K, N};
    Mat C{M, N};


    hpc::mm(1.0, A, B, -2.0, C, 1, 2);

    return 0;
}
