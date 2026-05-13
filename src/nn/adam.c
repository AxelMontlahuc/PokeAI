#include <stdlib.h>
#include <math.h>

#include "adam.h"
#include "config.h"

void optimizer_step_vector(Optimizer* optim, double param[MAX_OUTPUT_SIZE], double m[MAX_OUTPUT_SIZE], double v[MAX_OUTPUT_SIZE], double dL_dparam[MAX_OUTPUT_SIZE], int size) {
    for (int i=0; i<size; i++) {
        m[i] = optim->beta1 * m[i] + (1 - optim->beta1) * dL_dparam[i]; // Premier moment
        v[i] = optim->beta2 * v[i] + (1 - optim->beta2) * dL_dparam[i] * dL_dparam[i]; // Second moment
        double m_hat = m[i] / (1 - pow(optim->beta1, optim->t)); // Correction du biais pour le premier moment
        double v_hat = v[i] / (1 - pow(optim->beta2, optim->t)); // Correction du biais pour le second moment
        param[i] = param[i] - optim->lr * m_hat / (sqrt(v_hat) + optim->epsilon); // Mise à jour du paramètre
    }
}

void optimizer_step_matrix(Optimizer* optim, double param[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double m[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double v[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_dparam[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int rows, int cols) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            m[i][j] = optim->beta1 * m[i][j] + (1 - optim->beta1) * dL_dparam[i][j]; // Premier moment
            v[i][j] = optim->beta2 * v[i][j] + (1 - optim->beta2) * dL_dparam[i][j] * dL_dparam[i][j]; // Second moment
            double m_hat = m[i][j] / (1 - pow(optim->beta1, optim->t)); // Correction du biais pour le premier moment
            double v_hat = v[i][j] / (1 - pow(optim->beta2, optim->t)); // Correction du biais pour le second moment
            param[i][j] = param[i][j] - optim->lr * m_hat / (sqrt(v_hat) + optim->epsilon); // Mise à jour du paramètre
        }
    }
}