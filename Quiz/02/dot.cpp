#include <cassert>
#include <cstdlib>
#include "dot.hpp"
#include "vector.hpp"

double dot(const Vector &x, const Vector &y) {
    assert(x.n == y.n);
    double res = 0;
    for (size_t c = 0; c < x.n; ++c) {
        res += x(c) * y(c);
    }
    return res;
}
