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

#define ACTION_COUNT 7

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