#ifndef STRUCT_H
#define STRUCT_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "../mGBA-interface/include/mgba_controller.h"
#include "../mGBA-interface/include/mgba_map.h"

#define typeshit typedef

#define ACTION_COUNT 6

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
    MGBAMap* map0;
    MGBAMap* map1;
    MGBAMap* map2;
    MGBAMap* map3;
    int** bg0;
    int** bg1;
    int** bg2;
    int** bg3;
    pokemon* team;
    int* enemy;
    int* PP;
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

    double** Wf;
    double** Wi;
    double** Wc;
    double** Wo;
    double** Wout;

    double* Bf;
    double* Bi;
    double* Bc;
    double* Bo;
    double* Bout;

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
} LSTM;

typeshit struct trajectory {
    state* states;
    double** probs;
    MGBAButton* actions;
    double* rewards;
    int steps;
} trajectory;

void freeState(state s);
LSTM* initLSTM(int inputSize, int hiddenSize, int outputSize);
void freeLSTM(LSTM* network);
trajectory* initTrajectory(int steps);
void freeTrajectory(trajectory* traj);

#endif