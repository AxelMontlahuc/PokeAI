#ifndef ADAM_H
#define ADAM_H

#include "config.h"

typedef struct Optimizer Optimizer;

struct Optimizer {
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;          // instant t
};

void optimizer_step_vector_1(Optimizer* optim, double param[MAX_OUTPUT_SIZE], double m[MAX_OUTPUT_SIZE], double v[MAX_OUTPUT_SIZE], double dL_dparam[MAX_OUTPUT_SIZE], int size);
void optimizer_step_matrix_1(Optimizer* optim, double param[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double m[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double v[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_dparam[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int rows, int cols);
void optimizer_step_vector_2(Optimizer* optim, double param[HIDDEN_SIZE], double m[HIDDEN_SIZE], double v[HIDDEN_SIZE], double dL_dparam[HIDDEN_SIZE], int size);
void optimizer_step_matrix_2(Optimizer* optim, double param[HIDDEN_SIZE][COL_SIZE], double m[HIDDEN_SIZE][COL_SIZE], double v[HIDDEN_SIZE][COL_SIZE], double dL_dparam[HIDDEN_SIZE][COL_SIZE], int rows, int cols);

#endif