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

double* discountedPNL(double* rewards, double gamma, int steps) {
    double* G = calloc(steps, sizeof(double));
    assert(G != NULL);
    
    G[steps - 1] = rewards[steps - 1];
    for (int t = steps - 2; t >= 0; t--) {
        G[t] = rewards[t] + gamma * G[t+1];
    }

    return G;
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

void compute_gae(
    const double* rewards,
    const double* values,
    int steps,
    double gamma,
    double gae_lambda,
    double* out_advantages,
    double* out_returns
) {
    double gae = 0.0;
    for (int t = steps - 1; t >= 0; t--) {
        double v_t = values[t];
        double v_tp1 = (t + 1 < steps) ? values[t + 1] : 0.0;
        double delta = rewards[t] + gamma * v_tp1 - v_t;
        gae = delta + gamma * gae_lambda * gae;
        out_advantages[t] = gae;
        out_returns[t] = out_advantages[t] + v_t;
    }
}