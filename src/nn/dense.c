#include <stdlib.h>
#include <math.h>

#include "dense.h"
#include "config.h"

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
void init_dense(Dense* dense, int input_size, int output_size) {
    dense->input_size = input_size;
    dense->output_size = output_size;

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            dense->w_m[i][j] = 0.0;
            dense->w_v[i][j] = 0.0;
        }
    }

    xavier_init((double**)dense->w, input_size, output_size);

    for (int i = 0; i < output_size; i++) {
        dense->b[i] = 0.0;
        dense->b_m[i] = 0.0;
        dense->b_v[i] = 0.0;
    }
}

// Propagation
void dense_forward(Dense* dense, double* input, double* logits) {
    // Multiplication matricielle : output = w * input + b
    for (int i=0; i<dense->output_size; i++) {
        logits[i] = dense->b[i];
        for (int j=0; j<dense->input_size; j++) {
            logits[i] += dense->w[i][j] * input[j];
        }
    }
}

// Rétropropagation
void dense_backward(Dense* dense, double** input, int batch_size, double** dL_dlogits, double** dL_dw, double* dL_db, double** dL_dinput) {
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
    for (int k=0; k<batch_size; k++) {
        for (int j=0; j<dense->input_size; j++) {
            dL_dinput[k][j] = 0.0;
        }
        for (int j=0; j<dense->input_size; j++) {
            for (int i=0; i<dense->output_size; i++) {
                dL_dinput[k][j] += dL_dlogits[k][i] * dense->w[i][j];
            }
        }
    }
}