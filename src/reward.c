#include "reward.h"

double pnl(state s, state s_next) {
    double pnl = -0.01;

    if (s_next.clock == 80 && s.clock != 80) {
        pnl += 100;
    }
    for (int i = 0; i < 6; i += 1){
        if (s_next.team[i].level > s.team[i].level){
            pnl += 5;
        }
        if (s_next.team[i].HP <= s.team[i].HP/3) {
            pnl --;
        }
    }

    return pnl;
}