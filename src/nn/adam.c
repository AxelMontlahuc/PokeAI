#include <stdlib.h>
#include <math.h>

#include "adam.h"
#include "config.h"

void optimizer_step_vector_1(Optimizer* optim, float param[MAX_OUTPUT_SIZE], float m[MAX_OUTPUT_SIZE], float v[MAX_OUTPUT_SIZE], float dL_dparam[MAX_OUTPUT_SIZE], int size) {
    for (int i=0; i<size; i++) {
        m[i] = optim->beta1 * m[i] + (1.0f - optim->beta1) * dL_dparam[i]; // Premier moment
        v[i] = optim->beta2 * v[i] + (1.0f - optim->beta2) * dL_dparam[i] * dL_dparam[i]; // Second moment
        float m_hat = m[i] / (1.0f - powf(optim->beta1, (float)optim->t)); // Correction du biais pour le premier moment
        float v_hat = v[i] / (1.0f - powf(optim->beta2, (float)optim->t)); // Correction du biais pour le second moment
        param[i] = param[i] - optim->lr * m_hat / (sqrtf(v_hat) + optim->epsilon); // Mise à jour du paramètre
    }
}

void optimizer_step_matrix_1(Optimizer* optim, float param[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float m[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float v[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float dL_dparam[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int rows, int cols) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            m[i][j] = optim->beta1 * m[i][j] + (1.0f - optim->beta1) * dL_dparam[i][j]; // Premier moment
            v[i][j] = optim->beta2 * v[i][j] + (1.0f - optim->beta2) * dL_dparam[i][j] * dL_dparam[i][j]; // Second moment
            float m_hat = m[i][j] / (1.0f - powf(optim->beta1, (float)optim->t)); // Correction du biais pour le premier moment
            float v_hat = v[i][j] / (1.0f - powf(optim->beta2, (float)optim->t)); // Correction du biais pour le second moment
            param[i][j] = param[i][j] - optim->lr * m_hat / (sqrtf(v_hat) + optim->epsilon); // Mise à jour du paramètre
        }
    }
}

void optimizer_step_vector_2(Optimizer* optim, float param[HIDDEN_SIZE], float m[HIDDEN_SIZE], float v[HIDDEN_SIZE], float dL_dparam[HIDDEN_SIZE], int size) {
    for (int i=0; i<size; i++) {
        m[i] = optim->beta1 * m[i] + (1.0f - optim->beta1) * dL_dparam[i]; // Premier moment
        v[i] = optim->beta2 * v[i] + (1.0f - optim->beta2) * dL_dparam[i] * dL_dparam[i]; // Second moment
        float m_hat = m[i] / (1.0f - powf(optim->beta1, (float)optim->t)); // Correction du biais pour le premier moment
        float v_hat = v[i] / (1.0f - powf(optim->beta2, (float)optim->t)); // Correction du biais pour le second moment
        param[i] = param[i] - optim->lr * m_hat / (sqrtf(v_hat) + optim->epsilon); // Mise à jour du paramètre
    }
}

void optimizer_step_matrix_2(Optimizer* optim, float param[HIDDEN_SIZE][COL_SIZE], float m[HIDDEN_SIZE][COL_SIZE], float v[HIDDEN_SIZE][COL_SIZE], float dL_dparam[HIDDEN_SIZE][COL_SIZE], int rows, int cols) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            m[i][j] = optim->beta1 * m[i][j] + (1.0f - optim->beta1) * dL_dparam[i][j]; // Premier moment
            v[i][j] = optim->beta2 * v[i][j] + (1.0f - optim->beta2) * dL_dparam[i][j] * dL_dparam[i][j]; // Second moment
            float m_hat = m[i][j] / (1.0f - powf(optim->beta1, (float)optim->t)); // Correction du biais pour le premier moment
            float v_hat = v[i][j] / (1.0f - powf(optim->beta2, (float)optim->t)); // Correction du biais pour le second moment
            param[i][j] = param[i][j] - optim->lr * m_hat / (sqrtf(v_hat) + optim->epsilon); // Mise à jour du paramètre
        }
    }
}