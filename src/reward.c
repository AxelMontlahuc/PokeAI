#include "reward.h"

//bool HOUSE_FLAG = false;
//bool ROOM_FLAG = false;
//bool CLOCK_FLAG = false;
bool OUTDOOR_FLAG = false;
//bool OPP_HOUSE_FLAG = false;
//bool OPP_ROOM_FLAG = false;
bool ROUTE_101_FLAG = false;
//bool LAB_FLAG = false;
//bool LITTLEROOT_FLAG = false;
bool OLDAGE_FLAG = false;

double pnl(state s, state s_next) {
    double pnl = -0.005;

    /*if (!HOUSE_FLAG && s_next.zone == 1) {
        HOUSE_FLAG = true;
        pnl += 1.5;
    }

    if (!ROOM_FLAG && s_next.zone == 257) {
        ROOM_FLAG = true;
        pnl += 1.7;
    }

    if (!CLOCK_FLAG && s_next.clock == 80) {
        CLOCK_FLAG = true;
        pnl += 2.0;
    }

    if (CLOCK_FLAG && !OUTDOOR_FLAG && s_next.zone == 2304) {
        OUTDOOR_FLAG = true;
        pnl += 1.5;
    }

    if (!OPP_HOUSE_FLAG && s_next.zone == 513) {
        OPP_HOUSE_FLAG = true;
        pnl += 2.2;
    }

    if (!OPP_ROOM_FLAG && OPP_HOUSE_FLAG && s.zone == 513 && s_next.zone == 2304) {
        pnl -= 3.0;
    }

    if (!OPP_ROOM_FLAG && OPP_HOUSE_FLAG && s.zone == 2304 && s_next.zone == 513) {
        pnl += 2.2;
    }

    if (!OPP_ROOM_FLAG && s_next.zone == 769) {
        OPP_ROOM_FLAG = true;
        pnl += 2.0;
    }

    if (!OPP_HOUSE_FLAG && s_next.zone == 513) {
        OPP_HOUSE_FLAG = true;
        pnl += 1.0;
    }

    if (!OUTDOOR_FLAG && s_next.zone == 2304) {
        OUTDOOR_FLAG = true;
        pnl += 1.2;
    } */

    /*if (s.zone == 2304 && s_next.zone == 513) {
        pnl -= -0.6;
    }

    if (s.zone == 513 && s_next.zone == 2304) {
        pnl += 0.4;
    } */

    /*if (!ROUTE_101_FLAG && s_next.zone == 4096) {
        ROUTE_101_FLAG = true;
        pnl += 1.8;
    }

    if (ROUTE_101_FLAG && !LAB_FLAG && s_next.zone == 1025) {
        LAB_FLAG = true;
        pnl += 2.2;
    }

    if (!OUTDOOR_FLAG && s_next.zone == 2304) {
        OUTDOOR_FLAG = true;
        pnl += 1.6;
    }

    if (!ROUTE_101_FLAG && s_next.zone == 4096) {
        ROUTE_101_FLAG = true;
        pnl += 2.1;
    }

    if (ROUTE_101_FLAG && !LAB_FLAG && s_next.zone == 1025) {
        LAB_FLAG = true;
        pnl += 1.8;
    }*/

    if (!OLDAGE_FLAG && s_next.zone == 2560) {
        OLDAGE_FLAG = true;
        pnl += 4.0;
    }

    int maxHP_sum = 0;
    int HP_sum = 0;

    for (int i = 0; i < 6; i++) {
        maxHP_sum += s.team[i].maxHP;
        HP_sum += s.team[i].HP;

        if (s_next.team[i].level > s.team[i].level) {
            pnl += 2.5 * (double)(s_next.team[i].level - s.team[i].level);
        }
        if (s.team[i].HP == 0 && s_next.team[i].HP > 0) {
            pnl += 1.5;
        }

        if (s.team[i].HP > 0 && s_next.team[i].HP == 0) {
            pnl -= 2.0;
        }
    }

    if (HP_sum < maxHP_sum && s_next.zone == 514) {
        pnl += 0.5;
    }

    if (HP_sum < maxHP_sum && s.zone == 514 && s_next.zone == 2560) {
        pnl -= 0.8;
    }

    if (s.enemy[1] > 0 && s_next.enemy[1] == 0) {
        pnl += 1.0;
    }

    return pnl;
}

void normPNL(double* G, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += G[i];
    mean /= (double)n;

    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double d = G[i] - mean;
        variance += d * d;
    }
    variance /= (double)n;

    double std_deviation = sqrt(variance) + STD_EPS;

    for (int i = 0; i < n; i++) G[i] = (G[i] - mean) / std_deviation;
}

void computeGAE(double* rewards, double* values, int steps, double gamma, double lambda, double* out_advantages, double* out_returns) {
    double gae = 0.0;

    for (int t=steps-1; t>=0; t--) {
        double v = values[t];
        double v_next = (t + 1 < steps) ? values[t+1] : 0.0;

        double delta = rewards[t] + gamma * v_next - v;

        gae = delta + gamma * lambda * gae;

        out_advantages[t] = gae;
        out_returns[t] = out_advantages[t] + v;
    }
}

void resetFlags() {
    //HOUSE_FLAG = false;
    //ROOM_FLAG = false;
    //CLOCK_FLAG = false;
    OUTDOOR_FLAG = false;
    //OPP_HOUSE_FLAG = false;
    //OPP_ROOM_FLAG = false;
    ROUTE_101_FLAG = false;
    //LITTLEROOT_FLAG = false;
    //LAB_FLAG = false;
}

bool stopCondition() {
    return false;
}

bool saveCondition() {
    return OLDAGE_FLAG;
}