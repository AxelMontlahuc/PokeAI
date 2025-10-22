#include "reward.h"

bool HOUSE_FLAG = false;
bool ROOM_FLAG = false;
bool CLOCK_FLAG = false;
bool OUTDOOR_FLAG = false;
bool OPP_HOUSE_FLAG = false;
bool OPP_ROOM_FLAG = false;

double pnl(state s, state s_next) {
    double pnl = -0.005;

    if (!HOUSE_FLAG && s_next.zone == 1) {
        HOUSE_FLAG = true;
        pnl += 0.5;
    }

    if (!ROOM_FLAG && s_next.zone == 257) {
        ROOM_FLAG = true;
        pnl += 2.5;
    }

    if (!CLOCK_FLAG && s_next.clock == 80) {
        CLOCK_FLAG = true;
        pnl += 10.0;
    }

    if (CLOCK_FLAG && !OUTDOOR_FLAG && s_next.zone == 0) {
        OUTDOOR_FLAG = true;
        pnl += 1.0;
    }

    if (!OPP_HOUSE_FLAG && s_next.zone == 513) {
        OPP_HOUSE_FLAG = true;
        pnl += 7.5;
    }

    if (!OPP_ROOM_FLAG && OPP_HOUSE_FLAG && s.zone == 513 && s_next.zone == 0) {
        pnl -= 3.0;
    }

    if (!OPP_ROOM_FLAG && OPP_HOUSE_FLAG && s.zone == 0 && s_next.zone == 513) {
        pnl += 2.5;
    }

    if (!OPP_ROOM_FLAG && s_next.zone == 769) {
        OPP_ROOM_FLAG = true;
        pnl += 5.0;
    }

    return pnl;
}

void normPNL(double* G, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += G[i];
    mean /= (double)n;

    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = G[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    double std = sqrt(var) + 1e-8;

    for (int i = 0; i < n; i++) G[i] = (G[i] - mean) / std;
}

bool stop() {
    return CLOCK_FLAG;
}

void reset_flags() {
    HOUSE_FLAG = false;
    ROOM_FLAG = false;
    CLOCK_FLAG = false;
    OUTDOOR_FLAG = false;
    OPP_HOUSE_FLAG = false;
    OPP_ROOM_FLAG = false;
}