#ifndef ADAM_H
#define ADAM_H

#include "config.h"

typedef struct Optimizer Optimizer;

struct Optimizer {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    int t;          // instant t
};

void optimizer_step_vector_1(Optimizer* optim, float param[MAX_OUTPUT_SIZE], float m[MAX_OUTPUT_SIZE], float v[MAX_OUTPUT_SIZE], float dL_dparam[MAX_OUTPUT_SIZE], int size);
void optimizer_step_matrix_1(Optimizer* optim, float param[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float m[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float v[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float dL_dparam[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int rows, int cols);
void optimizer_step_vector_2(Optimizer* optim, float param[HIDDEN_SIZE], float m[HIDDEN_SIZE], float v[HIDDEN_SIZE], float dL_dparam[HIDDEN_SIZE], int size);
void optimizer_step_matrix_2(Optimizer* optim, float param[HIDDEN_SIZE][COL_SIZE], float m[HIDDEN_SIZE][COL_SIZE], float v[HIDDEN_SIZE][COL_SIZE], float dL_dparam[HIDDEN_SIZE][COL_SIZE], int rows, int cols);

#endif