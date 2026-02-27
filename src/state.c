#include "state.h"

state fetchState() {
    state s;
    
    int team_raw[6*8];

    gba_state(team_raw, s.enemy, s.PP, &s.zone, &s.clock, s.behavior, &s.player_x, &s.player_y);
    s.exploration = 1;

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

state fetchMGBAState(MGBAConnection conn) {
    state s;
    
    int team_raw[6*8];

    // Read behavior map via socket
    read_behavior_map(conn.sock, s.behavior, &s.player_x, &s.player_y);
    s.exploration = 1;

    // Read other state data
    int bg0[32][32];
    int bg2[32][32];
    read_state(conn.sock, team_raw, s.enemy, s.PP, &s.zone, &s.clock, bg0, bg2);

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

void convertState(state s, double* out) {
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

    for (int r = 0; r < 11; r++) {
        for (int c = 0; c < 11; c++) {
            int b = s.behavior[r][c];
            // Map integer codes to neural network inputs:
            // -1 (solid) -> -1.0, 0 (walkable) -> 0.0, 1 (grass) -> 0.5, 2 (interactable) -> 1.0
            double val;
            switch (b) {
                case -1: val = -1.0; break;
                case  0: val =  0.0; break;
                case  1: val =  0.5; break;
                case  2: val =  1.0; break;
                default: val =  0.0; break;
            }
            out[6*8 + 9 + r*11 + c] = val;
        }
    }
}

trajectory* initTrajectory(int steps) {
    trajectory* traj = malloc(sizeof(trajectory));
    assert(traj != NULL);

    traj->states = malloc(steps * sizeof(state));
    traj->actions = malloc(steps * sizeof(int));
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