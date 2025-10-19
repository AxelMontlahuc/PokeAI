#include "reward.h"

bool ROOM_FLAG = false;
bool CLOCK_FLAG = false;
bool OUTDOOR_FLAG = false;
bool OPP_HOUSE_FLAG = false;
bool OPP_ROOM_FLAG = false;

double pnl(state s, state s_next) {
    double pnl = -0.01;

    if (!CLOCK_FLAG && s_next.clock == 80) {
        CLOCK_FLAG = true;
        pnl += 50;
    }

    for (int i = 0; i < 6; i += 1) {
        if (s_next.team[i].level > s.team[i].level) {
            pnl += 5;
        }
        if (s_next.team[i].HP <= s.team[i].HP/3) {
            pnl --;
        }
    }

    return pnl;
}