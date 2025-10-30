#ifndef STATE_H
#define STATE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "../gba/gba.h"
#include "../mgba/include/mgba_connection.h"
#include "../mgba/include/mgba_controller.h"
#include "../mgba/include/mgba_intel.h"

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

typeshit struct trajectory {
    state* states;
    double** probs;
    int* actions;
    double* rewards;
    double* values;
    int steps;
} trajectory;

state fetchState();
state fetchMGBAState(MGBAConnection conn);
void convertState(state s, double* out);
trajectory* initTrajectory(int steps);
void freeTrajectory(trajectory* traj);

#endif