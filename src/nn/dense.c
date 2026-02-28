#include <stdlib.h>
#include <math.h>

#include "dense.h"

// Initialisation de Xavier : distribution uniforme dans [-limit, limit] avec limit = sqrt(6 / (input_size + output_size))
void xavier_init(double** matrix, int input_size, int output_size) {
    double limit = sqrt(6.0 / (input_size + output_size));

    for (int i=0; i<output_size; i++) {
        for (int j=0; j<input_size; j++) {
            double r = (double)rand() / RAND_MAX;
            double val = (r * 2.0 - 1.0) * limit;
            matrix[i][j] = val;
        }
    }
}

// Initialisation d'une couche dense
Dense* init_dense(int input_size, int output_size) {
    Dense* dense = malloc(sizeof(Dense));

    dense->input_size = input_size;
    dense->output_size = output_size;

    dense->w = malloc(output_size * sizeof(double*));
    dense->w_m = malloc(output_size * sizeof(double*));
    dense->w_v = malloc(output_size * sizeof(double*));

    for (int i=0; i<output_size; i++) {
        dense->w[i] = malloc(input_size * sizeof(double));
        dense->w_m[i] = calloc(input_size, sizeof(double));
        dense->w_v[i] = calloc(input_size, sizeof(double));
    }

    xavier_init(dense->w, input_size, output_size);

    dense->b = calloc(output_size, sizeof(double));
    dense->b_m = calloc(output_size, sizeof(double));
    dense->b_v = calloc(output_size, sizeof(double));

    return dense;
}

// Libération de la mémoire d'une couche dense
void free_dense(Dense* dense) {
    for (int i=0; i<dense->output_size; i++) {
        free(dense->w[i]);
        free(dense->w_m[i]);
        free(dense->w_v[i]);
    }

    free(dense->w);
    free(dense->w_m);
    free(dense->w_v);

    free(dense->b);
    free(dense->b_m);
    free(dense->b_v);

    free(dense);
}