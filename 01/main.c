#include <stdio.h>

#include "vector.h"
#include "walltime.h"


double vector_test_array[8];

void static_vector_test() {
    printf("Single vector:\n");
    init_vector(vector_test_array, 8, 1);
    print_vector(vector_test_array, 8, 1);

    printf("\n\nSeparate vectors:\n");
    init_vector(vector_test_array, 4, 1);
    init_vector(vector_test_array + 4, 4, 1);
    print_vector(vector_test_array, 4, 1);
    print_vector(vector_test_array + 4, 4, 1);

    printf("\n\nInterleaving vectors:\n");
    init_vector(vector_test_array, 4, 2);
    init_vector(vector_test_array + 1, 4, 2);
    print_vector(vector_test_array, 4, 2);
    print_vector(vector_test_array + 1, 4, 2);
}

void dynamic_vector_test() {
    size_t len;
    do {
        printf("\n\nEnter the length:");
    } while (scanf("%zu", &len) != 1);

    double *vec = (double*)malloc(len * sizeof(double));
    if (!vec) {
        printf("Malloc failed!");
        return;
    }

    printf("Single vector:\n");
    init_vector(vec, len, 1);
    print_vector(vec, len, 1);

    printf("\n\nSeparate vectors:\n");
    init_vector(vec, len/2, 1);
    init_vector(vec + len/2, len - len/2, 1);
    print_vector(vec, len/2, 1);
    print_vector(vec + len/2, len - len/2, 1);

    printf("\n\nInterleaving vectors:\n");
    init_vector(vec, len - len/2, 2);
    init_vector(vec+ 1, len/2, 2);
    print_vector(vec, len - len/2, 2);
    print_vector(vec+ 1, len/2, 2);

    free(vec);
}

void walltime_test() {
    printf("\n\n");
    for (size_t len = 8192; len <= 67108864; len *= 2) {
        double *vec_consec = (double*)malloc(len * sizeof(double) * 2);
        double *vec_interleaved = (double*)malloc(len * sizeof(double) * 2);

        double consec_start = walltime();
        init_vector(vec_consec, len, 1);
        init_vector(vec_consec + len, len, 1);
        double consec_time = walltime() - consec_start;

        double interleaved_start = walltime();
        init_vector(vec_interleaved, len, 2);
        init_vector(vec_interleaved + 1, len, 2);
        double interleaved_time = walltime() - consec_start;

        printf("%8zu\t%lf\t%lf\n", len, consec_time, interleaved_time);

        free(vec_consec);
        free(vec_interleaved);
    }
}


int main() {
    static_vector_test();
    dynamic_vector_test();
    walltime_test();
    return 0;
}
