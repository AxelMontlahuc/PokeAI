#ifndef ADAM_H
#define ADAM_H

typedef struct Optimizer Optimizer;

struct Optimizer {
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;          // instant t
};

void optimizer_step_vector(Optimizer* optim, double* param, double* m, double* v, double* dL_dparam, int size);
void optimizer_step_matrix(Optimizer* optim, double** param, double** m, double** v, double** dL_dparam, int rows, int cols);

#endif