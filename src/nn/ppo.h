#ifndef PPO_H
#define PPO_H

#include "dense.h"
#include "config.h"

struct Trajectory {
    int states[TRAJ_SIZE][INPUT_SIZE];
    double hidden_states[TRAJ_SIZE][HIDDEN_SIZE];
    double z[TRAJ_SIZE][COL_SIZE];    // Concaténation de l'entrée et de l'état caché précédent pour le LSTM (i.e. [x_t; h_{t-1}])
    double f[TRAJ_SIZE][HIDDEN_SIZE];
    double i[TRAJ_SIZE][HIDDEN_SIZE];
    double g[TRAJ_SIZE][HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    double o[TRAJ_SIZE][HIDDEN_SIZE];
    double c[TRAJ_SIZE][HIDDEN_SIZE]; // Mémoire à long terme (cellule) du LSTM
    int actions[TRAJ_SIZE];
    double rewards[TRAJ_SIZE];
    double probs[TRAJ_SIZE][POLICY_OUTPUT_SIZE];
    double values[TRAJ_SIZE];
    double advantages[TRAJ_SIZE];
    int done[TRAJ_SIZE]; // Permet de faire les calculs de GAE correctement quand l'épisode termine prématurément
};
typedef struct Trajectory Trajectory;

struct Minibatch {
    int states[MINIBATCH_SIZE][INPUT_SIZE];
    double hidden_states[MINIBATCH_SIZE][HIDDEN_SIZE];
    double z[MINIBATCH_SIZE][COL_SIZE];    // Concaténation de l'entrée et de l'état caché précédent pour le LSTM (i.e. [x_t; h_{t-1}])
    double f[MINIBATCH_SIZE][HIDDEN_SIZE];
    double i[MINIBATCH_SIZE][HIDDEN_SIZE];
    double g[MINIBATCH_SIZE][HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    double o[MINIBATCH_SIZE][HIDDEN_SIZE];
    double c[MINIBATCH_SIZE][HIDDEN_SIZE]; // Mémoire à long terme (cellule) du LSTM
    double c_ini[HIDDEN_SIZE];             // Requis pour la rétropropagation du LSTM
    int actions[MINIBATCH_SIZE];
    double rewards[MINIBATCH_SIZE];
    double probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE];
    double values[MINIBATCH_SIZE];
    double ppo_loss;
    double value_loss;
    double advantages[MINIBATCH_SIZE];
    double ratios[MINIBATCH_SIZE];
    double unclipped_ratios[MINIBATCH_SIZE];
    int clipped[MINIBATCH_SIZE];
    double kl[MINIBATCH_SIZE];
    int done[MINIBATCH_SIZE]; // Permet de faire les calculs de GAE correctement quand l'épisode termine prématurément
};
typedef struct Minibatch Minibatch;

double value_backward(Dense* value_head, double pred[MINIBATCH_SIZE], double target[MINIBATCH_SIZE], double input[MINIBATCH_SIZE][HIDDEN_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE]);
void compute_advantages(double rewards[TRAJ_SIZE], double values[TRAJ_SIZE], int done[TRAJ_SIZE], double advantages[TRAJ_SIZE]);
double ppo_loss(Minibatch* minibatch, double prob[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], double old_prob[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], double advantages[MINIBATCH_SIZE], int actions[MINIBATCH_SIZE], double dlogp[MINIBATCH_SIZE], double ent_coeff);
void policy_backward(Dense* policy_head, Minibatch* minibatch, double new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE], double ent_coeff);

#endif