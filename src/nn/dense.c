#include <stdlib.h>
#include <math.h>

#include "dense.h"
#include "config.h"

// Initialisation de Xavier : distribution uniforme dans [-limit, limit] avec limit = sqrt(6 / (input_size + output_size))
static void xavier_init_dense(float matrix[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int input_size, int output_size) {
    float limit = sqrtf(6.0f / (float)(input_size + output_size));

    for (int i=0; i<output_size; i++) {
        for (int j=0; j<input_size; j++) {
            float r = (float)rand() / (float)RAND_MAX;
            float val = (r * 2.0f - 1.0f) * limit;
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

    xavier_init_dense(dense->w, input_size, output_size);

    for (int i = 0; i < output_size; i++) {
        dense->b[i] = 0.0;
        dense->b_m[i] = 0.0;
        dense->b_v[i] = 0.0;
    }
}

// Propagation
void dense_forward(Dense* dense, float* input, float* logits) {
    // Multiplication matricielle : output = w * input + b
    for (int i=0; i<dense->output_size; i++) {
        logits[i] = dense->b[i];
        for (int j=0; j<dense->input_size; j++) {
            logits[i] += dense->w[i][j] * input[j];
        }
    }
}

// Rétropropagation
void dense_backward(Dense* dense, float input[MINIBATCH_SIZE][HIDDEN_SIZE], float dL_dlogits[MINIBATCH_SIZE][MAX_OUTPUT_SIZE], float dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float dL_db[MAX_OUTPUT_SIZE], float dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE]) {
    // Gradients pour les poids
    for (int i=0; i<dense->output_size; i++) {
        for (int j=0; j<dense->input_size; j++) {
            // Calcul du gradient moyen sur le batch
            float grad = 0.0f;
            for (int k=0; k<MINIBATCH_SIZE; k++) {
                grad += dL_dlogits[k][i] * input[k][j];
            }
            grad /= (float)MINIBATCH_SIZE; // Moyenne sur le batch

            dL_dw[i][j] = grad;
        }
    }

    // Gradients pour les biais
    for (int i=0; i<dense->output_size; i++) {
        float grad = 0.0f;
        for (int k=0; k<MINIBATCH_SIZE; k++) {
            grad += dL_dlogits[k][i];
        }
        dL_db[i] = grad / (float)MINIBATCH_SIZE;
    }

    // Gradients pour l'entrée (nécessaire pour la rétropropagation dans les couches précédentes)
    for (int k=0; k<MINIBATCH_SIZE; k++) {
        for (int j=0; j<dense->input_size; j++) {
            dL_dinput[k][j] = 0.0f;
        }
        for (int j=0; j<dense->input_size; j++) {
            for (int i=0; i<dense->output_size; i++) {
                dL_dinput[k][j] += dL_dlogits[k][i] * dense->w[i][j];
            }
        }
    }
}