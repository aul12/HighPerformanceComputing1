/**
 * @file page01.c
 * @author paul
 * @date 02.11.20
 * Description here TODO
 */
#include "vector.h"

#include <stdio.h>

void init_vector(double *vec, size_t len, size_t stride) {
    for (size_t c=0; c<len; ++c) {
        vec[c * stride] = (double)(c+1U);
    }
}

void print_vector(const double *vec, size_t len, size_t stride) {
    printf("[");
    for (size_t c=0; c<len; ++c) {
        if (c != 0) {
            printf(", ");
        }
        printf("%f", vec[c * stride]);
    }
    printf("]");
}
