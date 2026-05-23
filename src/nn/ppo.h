#ifndef PPO_H
#define PPO_H

#include "dense.h"
#include "config.h"

struct Trajectory {
    int states[TRAJ_SIZE][INPUT_SIZE];
    float hidden_states[TRAJ_SIZE][HIDDEN_SIZE];
    float z[TRAJ_SIZE][COL_SIZE];    // Concaténation de l'entrée et de l'état caché précédent pour le LSTM (i.e. [x_t; h_{t-1}])
    float f[TRAJ_SIZE][HIDDEN_SIZE];
    float i[TRAJ_SIZE][HIDDEN_SIZE];
    float g[TRAJ_SIZE][HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    float o[TRAJ_SIZE][HIDDEN_SIZE];
    float c[TRAJ_SIZE][HIDDEN_SIZE]; // Mémoire à long terme (cellule) du LSTM
    int actions[TRAJ_SIZE];
    float rewards[TRAJ_SIZE];
    float probs[TRAJ_SIZE][POLICY_OUTPUT_SIZE];
    float values[TRAJ_SIZE];
    float advantages[TRAJ_SIZE];
    int done[TRAJ_SIZE]; // Permet de faire les calculs de GAE correctement quand l'épisode termine prématurément
};
typedef struct Trajectory Trajectory;

struct Minibatch {
    int states[MINIBATCH_SIZE][INPUT_SIZE];
    float hidden_states[MINIBATCH_SIZE][HIDDEN_SIZE];
    float z[MINIBATCH_SIZE][COL_SIZE];    // Concaténation de l'entrée et de l'état caché précédent pour le LSTM (i.e. [x_t; h_{t-1}])
    float f[MINIBATCH_SIZE][HIDDEN_SIZE];
    float i[MINIBATCH_SIZE][HIDDEN_SIZE];
    float g[MINIBATCH_SIZE][HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    float o[MINIBATCH_SIZE][HIDDEN_SIZE];
    float c[MINIBATCH_SIZE][HIDDEN_SIZE]; // Mémoire à long terme (cellule) du LSTM
    float c_ini[NUM_SEQS][HIDDEN_SIZE];
    float h_ini[NUM_SEQS][HIDDEN_SIZE];
    int actions[MINIBATCH_SIZE];
    float rewards[MINIBATCH_SIZE];
    float probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE];
    float values[MINIBATCH_SIZE];
    float ppo_loss;
    float value_loss;
    float advantages[MINIBATCH_SIZE];
    float ratios[MINIBATCH_SIZE];
    float unclipped_ratios[MINIBATCH_SIZE];
    int clipped[MINIBATCH_SIZE];
    float kl[MINIBATCH_SIZE];
    int done[MINIBATCH_SIZE]; // Permet de faire les calculs de GAE correctement quand l'épisode termine prématurément
};
typedef struct Minibatch Minibatch;

float value_backward(Dense* value_head, float pred[MINIBATCH_SIZE], float target[MINIBATCH_SIZE], float input[MINIBATCH_SIZE][HIDDEN_SIZE], float dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float dL_db[MAX_OUTPUT_SIZE], float dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE]);
void compute_advantages(float rewards[TRAJ_SIZE], float values[TRAJ_SIZE], int done[TRAJ_SIZE], float advantages[TRAJ_SIZE]);
float ppo_loss(Minibatch* minibatch, float prob[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], float old_prob[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], float advantages[MINIBATCH_SIZE], int actions[MINIBATCH_SIZE], float dlogp[MINIBATCH_SIZE], float ent_coeff);
void policy_backward(Dense* policy_head, Minibatch* minibatch, float new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], float dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], float dL_db[MAX_OUTPUT_SIZE], float dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE], float ent_coeff, float temperature);

#endif