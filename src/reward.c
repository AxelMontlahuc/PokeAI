#include "reward.h"

bool HOUSE_FLAG = false;
bool ROOM_FLAG = false;
bool CLOCK_FLAG = false;
bool OUTDOOR_FLAG = false;
bool OPP_HOUSE_FLAG = false;
bool OPP_ROOM_FLAG = false;

double pnl(state s, state s_next) {
    double pnl = -0.01;

    if (!HOUSE_FLAG && s_next.zone == 1) {
        HOUSE_FLAG = true;
        pnl += 0.5;
    }

    if (!ROOM_FLAG && s_next.zone == 101) {
        ROOM_FLAG = true;
        pnl += 2.0;
    }

    if (!CLOCK_FLAG && s.zone == 101 && s_next.zone == 1) {
        pnl -= 2.0;
    }

    if (!CLOCK_FLAG && ROOM_FLAG && s.zone == 1 && s_next.zone == 101) {
        pnl += 1.5;
    }

    if (!CLOCK_FLAG && s_next.clock == 80) {
        CLOCK_FLAG = true;
        pnl += 10.0;
    }

    if (!OUTDOOR_FLAG && s_next.zone == 0) {
        OUTDOOR_FLAG = true;
        pnl += 1.0;
    }

    if (!OPP_HOUSE_FLAG && s_next.zone == 201) {
        OPP_HOUSE_FLAG = true;
        pnl += 7.5;
    }

    if (!OPP_ROOM_FLAG && OPP_HOUSE_FLAG && s.zone == 201 && s_next.zone == 0) {
        pnl -= 3.0;
    }

    if (!OPP_ROOM_FLAG && OPP_HOUSE_FLAG && s.zone == 0 && s_next.zone == 201) {
        pnl += 2.5;
    }

    if (!OPP_ROOM_FLAG && s_next.zone == 301) {
        OPP_ROOM_FLAG = true;
        pnl += 5.0;
    }

    for (int i = 0; i < 6; i += 1) {
        if (s_next.team[i].level > s.team[i].level) {
            pnl += 5.0;
        }
        if (s_next.team[i].HP <= s.team[i].HP/3) {
            pnl -= 1.0;
        }
    }

    return pnl;
}

bool stop() {
    return CLOCK_FLAG;
}