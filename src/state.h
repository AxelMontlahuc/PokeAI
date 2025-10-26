#ifndef STATE_H
#define STATE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "../mGBA-interface/include/mgba_connection.h"
#include "../mGBA-interface/include/mgba_controller.h"
#include "../mGBA-interface/include/mgba_intel.h"

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

static inline MGBAButton indexToAction(int index) {
    switch (index) {
        case 0: return MGBA_BUTTON_UP;
        case 1: return MGBA_BUTTON_DOWN;
        case 2: return MGBA_BUTTON_LEFT;
        case 3: return MGBA_BUTTON_RIGHT;
        case 4: return MGBA_BUTTON_A;
        case 5: return MGBA_BUTTON_B;
        case 6: return MGBA_BUTTON_START;
        default: return MGBA_BUTTON_B;
    }
}

state fetchState(MGBAConnection conn);
double* convertState(state s);
trajectory* initTrajectory(int steps);
void freeTrajectory(trajectory* traj);

#endif