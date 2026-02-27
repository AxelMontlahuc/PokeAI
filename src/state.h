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
    int behavior[11][11];  // -1=solid, 0=walkable, 1=grass, 2=interactable
    int player_x;
    int player_y;
    pokemon team[6];
    int enemy[3];
    int PP[4];
    int exploration;
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