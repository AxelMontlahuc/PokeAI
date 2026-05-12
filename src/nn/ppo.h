#ifndef PPO_H
#define PPO_H

#include "config.h"

struct Trajectory {
    double states[BATCH_SIZE][INPUT_SIZE];
    double hidden_states[BATCH_SIZE][HIDDEN_SIZE];
    int actions[BATCH_SIZE];
    double rewards[BATCH_SIZE];
    double probs[BATCH_SIZE][POLICY_OUTPUT_SIZE];
    double values[BATCH_SIZE];
    int done[BATCH_SIZE]; // Permet de faire les calculs de GAE correctement quand l'épisode termine prématurément
};
typedef struct Trajectory Trajectory;

void value_backward(Dense* value_head, double pred[BATCH_SIZE], double target[BATCH_SIZE], double input[BATCH_SIZE][HIDDEN_SIZE], int batch_size, double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[BATCH_SIZE][HIDDEN_SIZE]);
void compute_advantages(double rewards[BATCH_SIZE], double values[BATCH_SIZE], int done[BATCH_SIZE], double advantages[BATCH_SIZE]);
double ppo_loss(double prob[BATCH_SIZE][POLICY_OUTPUT_SIZE], double old_prob[BATCH_SIZE][POLICY_OUTPUT_SIZE], double advantages[BATCH_SIZE], int actions[BATCH_SIZE], double dlogp[BATCH_SIZE]);
#endif