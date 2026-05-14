#ifndef PPO_H
#define PPO_H

#include "dense.h"
#include "config.h"

struct Trajectory {
    int states[BATCH_SIZE][INPUT_SIZE];
    double hidden_states[BATCH_SIZE][HIDDEN_SIZE];
    double z[BATCH_SIZE][COL_SIZE];    // Concaténation de l'entrée et de l'état caché précédent pour le LSTM (i.e. [x_t; h_{t-1}])
    double f[BATCH_SIZE][HIDDEN_SIZE];
    double i[BATCH_SIZE][HIDDEN_SIZE];
    double g[BATCH_SIZE][HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    double o[BATCH_SIZE][HIDDEN_SIZE];
    double c[BATCH_SIZE][HIDDEN_SIZE]; // Mémoire à long terme (cellule) du LSTM
    int actions[BATCH_SIZE];
    double rewards[BATCH_SIZE];
    double probs[BATCH_SIZE][POLICY_OUTPUT_SIZE];
    double values[BATCH_SIZE];
    double ppo_loss;
    double value_loss;
    double advantages[BATCH_SIZE];
    double ratios[BATCH_SIZE];
    double unclipped_ratios[BATCH_SIZE];
    int clipped[BATCH_SIZE];
    double kl[BATCH_SIZE];
    int done[BATCH_SIZE]; // Permet de faire les calculs de GAE correctement quand l'épisode termine prématurément
};
typedef struct Trajectory Trajectory;

double value_backward(Dense* value_head, double pred[BATCH_SIZE], double target[BATCH_SIZE], double input[BATCH_SIZE][HIDDEN_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[BATCH_SIZE][HIDDEN_SIZE]);
void compute_advantages(double rewards[BATCH_SIZE], double values[BATCH_SIZE], int done[BATCH_SIZE], double advantages[BATCH_SIZE]);
double ppo_loss(Trajectory* traj, double prob[BATCH_SIZE][POLICY_OUTPUT_SIZE], double old_prob[BATCH_SIZE][POLICY_OUTPUT_SIZE], double advantages[BATCH_SIZE], int actions[BATCH_SIZE], double dlogp[BATCH_SIZE]);
void policy_backward(Dense* policy_head, Trajectory* traj, double new_probs[BATCH_SIZE][POLICY_OUTPUT_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[BATCH_SIZE][HIDDEN_SIZE]);

#endif