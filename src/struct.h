#ifndef STRUCT_H
#define STRUCT_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "../mGBA-interface/include/mgba_controller.h"
#include "constants.h"

#define typeshit typedef

typeshit struct pokemon {
    int maxHP;
    int HP;
    int level;
    int ATK;
    int DEF;
    int SPEED;
    int ATK_SPE;
    int DEF_SPE;
} pokemon;

typeshit struct state {
    int bg0[32][32];
    int bg2[32][32];
    pokemon team[6];
    int enemy[3];
    int PP[4];
    int zone;
    int clock;
} state;

typeshit struct LSTM {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double* hiddenState;
    double* cellState;
    double* logits;
    double* probs;
    double last_value;

    double** Wf;
    double** Wi;
    double** Wc;
    double** Wo;
    double** Wout;
    double* Wv;

    double* Bf;
    double* Bi;
    double* Bc;
    double* Bo;
    double* Bout;
    double Bv;

    int adam_t;
    double** Wf_m; 
    double** Wf_v;
    double** Wi_m; 
    double** Wi_v;
    double** Wc_m; 
    double** Wc_v;
    double** Wo_m; 
    double** Wo_v;
    double** Wout_m; 
    double** Wout_v;
    double* Wv_m;
    double* Wv_v;

    double* Bf_m; 
    double* Bf_v;
    double* Bi_m; 
    double* Bi_v;
    double* Bc_m; 
    double* Bc_v;
    double* Bo_m; 
    double* Bo_v;
    double* Bout_m; 
    double* Bout_v;
    double Bv_m;
    double Bv_v;
} LSTM;

typeshit struct trajectory {
    state* states;
    double** probs;
    MGBAButton* actions;
    double* rewards;
    double* values;
    int steps;
} trajectory;

static inline int actionToIndex(MGBAButton action) {
    switch (action) {
        case MGBA_BUTTON_UP: return 0;
        case MGBA_BUTTON_DOWN: return 1;
        case MGBA_BUTTON_LEFT: return 2;
        case MGBA_BUTTON_RIGHT: return 3;
        case MGBA_BUTTON_A: return 4;
        case MGBA_BUTTON_B: return 5;
        case MGBA_BUTTON_START: return 6;
        default: return 5;
    }
}

LSTM* initLSTM(int inputSize, int hiddenSize, int outputSize);
void freeLSTM(LSTM* network);
trajectory* initTrajectory(int steps);
void freeTrajectory(trajectory* traj);

#endif