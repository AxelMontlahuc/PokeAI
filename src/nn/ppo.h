#ifndef PPO_H
#define PPO_H

#include "config.h"

struct TrajectoryBuffer {
    double states[BATCH_SIZE][INPUT_SIZE];
    int actions[BATCH_SIZE];
    double rewards[BATCH_SIZE];
    double logits[BATCH_SIZE][POLICY_OUTPUT_SIZE];
    double values[BATCH_SIZE][VALUE_OUTPUT_SIZE];
};
typedef struct TrajectoryBuffer TrajectoryBuffer;

void value_backward(Dense* value_head, double pred[BATCH_SIZE][VALUE_OUTPUT_SIZE], double target[BATCH_SIZE][VALUE_OUTPUT_SIZE], double input[BATCH_SIZE][INPUT_SIZE], int batch_size, double dL_dw[MAX_OUTPUT_SIZE][INPUT_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[BATCH_SIZE][INPUT_SIZE]);

#endif