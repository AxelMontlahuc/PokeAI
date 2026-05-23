#ifndef DENSE_H
#define DENSE_H

#include "config.h"

typedef struct Dense Dense;

struct Dense {
    int input_size;
    int output_size;

    float w[MAX_OUTPUT_SIZE][HIDDEN_SIZE];
    float w_m[MAX_OUTPUT_SIZE][HIDDEN_SIZE];
    float w_v[MAX_OUTPUT_SIZE][HIDDEN_SIZE];

    float b[MAX_OUTPUT_SIZE];
    float b_m[MAX_OUTPUT_SIZE];
    float b_v[MAX_OUTPUT_SIZE];
};

void init_dense(Dense* dense, int input_size, int output_size);
void dense_forward(Dense* dense, float* input, float* logits);
void dense_backward(Dense* dense,
    float input[MINIBATCH_SIZE][HIDDEN_SIZE],
    float dL_dlogits[MINIBATCH_SIZE][MAX_OUTPUT_SIZE],
    float dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE],
    float dL_db[MAX_OUTPUT_SIZE],
    float dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE]
);

#endif