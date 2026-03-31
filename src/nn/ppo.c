#include <stdlib.h>
#include <math.h>

#include "ppo.h"
#include "dense.h"
#include "config.h"

// Fonction coût de la value head/critic (MSE)
double value_loss(double* pred, double* target, int size) {
    double loss = 0;

    for (int i=0; i<size; i++) {
        double diff = pred[i] - target[i];
        loss += diff * diff;
    }
    loss /= size;

    return loss;
}

// Rétropropagation de la value head/critic
void value_backward(Dense* value_head, double** pred, double** target, double** input, int batch_size, double** dL_dw, double* dL_db, double** dL_dinput) {
    double dL_dlogits[BATCH_SIZE_MAX][VALUE_OUTPUT_SIZE];

    for (int k=0; k<batch_size; k++) {
        for (int i=0; i<value_head->output_size; i++) {
            dL_dlogits[k][i] = 2 * (pred[k][i] - target[k][i]) / batch_size; // dL/dlogits = 2 * (pred - target) / batch_size
        }
    }

    dense_backward(value_head, input, batch_size, (double**)dL_dlogits, dL_dw, dL_db, dL_dinput);
}