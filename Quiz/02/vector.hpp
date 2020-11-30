#ifndef HPC_VECTOR_HPP
#define HPC_VECTOR_HPP

#include <cassert>
#include <cstdlib>

struct Vector {
    // Similar to Matrix all as public members
    const std::size_t n;
    const std::ptrdiff_t inc;
    double *data;

    explicit Vector(std::size_t length) : n{length}, inc{1}, data{new double[n]()} {}

    // Rule of five
    Vector(const Vector &) = delete;

    auto operator=(const Vector &) -> Vector & = delete;

    Vector(Vector &&) = delete;

    auto operator=(Vector &&) -> Vector & = delete;

    ~Vector() {
        delete[] data;
    }

    const double &operator()(std::size_t i) const {
        assert(i < n);
        return data[i * inc];
    }

    double &operator()(std::size_t i) {
        assert(i < n);
        return data[i * inc];
    }

    void init() {
        for (size_t c = 0; c < n; ++c) {
            data[c * inc] = c;
        }
    }
};

#endif
