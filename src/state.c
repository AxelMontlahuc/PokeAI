#include "state.h"

state fetchState(MGBAConnection conn) {
    state s;
    
    int team_raw[6*8];

    read_state(conn.sock, team_raw, s.enemy, s.PP, &s.zone, &s.clock, s.bg0, s.bg2);

    for (int i = 0; i < 6; i++) {
        int o = i * 8;
        s.team[i].maxHP   = team_raw[o + 0];
        s.team[i].HP      = team_raw[o + 1];
        s.team[i].level   = team_raw[o + 2];
        s.team[i].ATK     = team_raw[o + 3];
        s.team[i].DEF     = team_raw[o + 4];
        s.team[i].SPEED   = team_raw[o + 5];
        s.team[i].ATK_SPE = team_raw[o + 6];
        s.team[i].DEF_SPE = team_raw[o + 7];
    }

    return s;
}

double* convertState(state s) {
    double* out = malloc((6*8 + 4 + 3 + 2 + 2*32*32) * sizeof(double));
    for (int i=0; i<6; i++) {
        out[i*8 + 0] = (double)s.team[i].maxHP / 300.0;
        out[i*8 + 1] = (double)s.team[i].HP / 300.0;
        out[i*8 + 2] = (double)s.team[i].level / 100.0;
        out[i*8 + 3] = (double)s.team[i].ATK / 300.0;
        out[i*8 + 4] = (double)s.team[i].DEF / 300.0;
        out[i*8 + 5] = (double)s.team[i].ATK_SPE / 300.0;
        out[i*8 + 6] = (double)s.team[i].DEF_SPE / 300.0;
        out[i*8 + 7] = (double)s.team[i].SPEED / 300.0;
    }
    out[6*8 + 0] = (double)s.PP[0] / 64.0;
    out[6*8 + 1] = (double)s.PP[1] / 64.0;
    out[6*8 + 2] = (double)s.PP[2] / 64.0;
    out[6*8 + 3] = (double)s.PP[3] / 64.0;
    out[6*8 + 4] = (double)s.enemy[0] / 300.0;
    out[6*8 + 5] = (double)s.enemy[1] / 300.0;
    out[6*8 + 6] = (double)s.enemy[2] / 100.0;
    out[6*8 + 7] = (double)s.zone / 255.0;
    out[6*8 + 8] = (double)s.clock / 255.0;

    for (int k=0; k<32; k++) {
        for (int l=0; l<32; l++) {
            out[6*8 + 9 + k*32 + l] = (double)s.bg0[k][l] / 2048.0 - 0.5;
        }
    }

    for (int k=0; k<32; k++) {
        for (int l=0; l<32; l++) {
            out[6*8 + 9 + 32*32 + k*32 + l] = (double)s.bg2[k][l] / 2048.0 - 0.5;
        }
    }

    return out;
}

trajectory* initTrajectory(int steps) {
    trajectory* traj = malloc(sizeof(trajectory));
    assert(traj != NULL);

    traj->states = malloc(steps * sizeof(state));
    traj->actions = malloc(steps * sizeof(MGBAButton));
    traj->rewards = malloc(steps * sizeof(double));
    traj->probs = malloc(steps * sizeof(double*));
    traj->values = malloc(steps * sizeof(double));
    assert(traj->states != NULL && traj->actions != NULL && traj->rewards != NULL && traj->probs != NULL && traj->values != NULL);

    for (int i=0; i<steps; i++) {
        traj->probs[i] = NULL;
    }

    traj->steps = steps;

    return traj;
}

void freeTrajectory(trajectory* traj) {
    free(traj->states);
    for (int i=0; i<traj->steps; i++) {
        free(traj->probs[i]);
    }
    free(traj->actions);
    free(traj->rewards);
    free(traj->probs);
    free(traj->values);
    free(traj);
}