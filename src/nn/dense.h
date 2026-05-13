#ifndef DENSE_H
#define DENSE_H

#include "config.h"

typedef struct Dense Dense;

struct Dense {
    int input_size;
    int output_size;

    double w[MAX_OUTPUT_SIZE][HIDDEN_SIZE];
    double w_m[MAX_OUTPUT_SIZE][HIDDEN_SIZE];
    double w_v[MAX_OUTPUT_SIZE][HIDDEN_SIZE];

    double b[MAX_OUTPUT_SIZE];
    double b_m[MAX_OUTPUT_SIZE];
    double b_v[MAX_OUTPUT_SIZE];
};

void xavier_init(double matrix[MAX_OUTPUT_SIZE][HIDDEN_SIZE], int input_size, int output_size);
void init_dense(Dense* dense, int input_size, int output_size);
void dense_forward(Dense* dense, double* input, double* logits);
void dense_backward(Dense* dense,
    double input[BATCH_SIZE][HIDDEN_SIZE],
    double dL_dlogits[BATCH_SIZE][MAX_OUTPUT_SIZE],
    double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE],
    double dL_db[MAX_OUTPUT_SIZE],
    double dL_dinput[BATCH_SIZE][HIDDEN_SIZE]
);

#endif