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

void optimizer_step_vector(Optimizer* optim, double param[MAX_OUTPUT_SIZE], double m[MAX_OUTPUT_SIZE], double v[MAX_OUTPUT_SIZE], double dL_dparam[MAX_OUTPUT_SIZE], int size);
void optimizer_step_matrix(Optimizer* optim, double param[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double m[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double v[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_dparam[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int rows, int cols);

#endif