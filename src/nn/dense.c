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

// Propagation
double* dense_forward(Dense* dense, double* input) {
    double* logits = malloc(dense->output_size * sizeof(double));

    // Multiplication matricielle : output = w * input + b
    for (int i=0; i<dense->output_size; i++) {
        logits[i] = dense->b[i];
        for (int j=0; j<dense->input_size; j++) {
            logits[i] += dense->w[i][j] * input[j];
        }
    }

    return logits;
}

// Rétropropagation
double** dense_backward(Dense* dense, double** input, int batch_size, double** dL_dlogits, double** dL_dw, double* dL_db) {
    // Gradients pour les poids
    for (int i=0; i<dense->output_size; i++) {
        for (int j=0; j<dense->input_size; j++) {
            // Calcul du gradient moyen sur le batch
            double grad = 0;
            for (int k=0; k<batch_size; k++) {
                grad += dL_dlogits[k][i] * input[k][j];
            }
            grad /= batch_size; // Moyenne sur le batch

            dL_dw[i][j] = grad;
        }
    }

    // Gradients pour les biais
    for (int i=0; i<dense->output_size; i++) {
        double grad = 0;
        for (int k=0; k<batch_size; k++) {
            grad += dL_dlogits[k][i];
        }
        dL_db[i] = grad / batch_size;
    }

    // Gradients pour l'entrée (nécessaire pour la rétropropagation dans les couches précédentes)
    double** dL_dinput = malloc(batch_size * sizeof(double*));
    for (int k=0; k<batch_size; k++) {
        dL_dinput[k] = calloc(dense->input_size, sizeof(double));
        for (int j=0; j<dense->input_size; j++) {
            for (int i=0; i<dense->output_size; i++) {
                dL_dinput[k][j] += dL_dlogits[k][i] * dense->w[i][j];
            }
        }
    }

    return dL_dinput;
}