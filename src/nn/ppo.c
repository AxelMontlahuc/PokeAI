#include <stdlib.h>
#include <math.h>

#include "ppo.h"
#include "dense.h"
#include "config.h"

// Fonction coût de la value head/critic (MSE)
double value_loss(double* pred, double* target, int size) {
    double loss = 0;

    for (int i=0; i<size; i++) {
        double diff = pred[i] - target[i];
        loss += diff * diff;
    }
    loss /= size;

    return loss;
}

// Rétropropagation de la value head/critic
void value_backward(Dense* value_head, double pred[BATCH_SIZE], double target[BATCH_SIZE], double input[BATCH_SIZE][HIDDEN_SIZE], int batch_size, double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[BATCH_SIZE][HIDDEN_SIZE]) {
    double dL_dlogits[BATCH_SIZE][MAX_OUTPUT_SIZE];

    for (int t=0; t<batch_size; t++) {
        dL_dlogits[t][0] = 2 * (pred[t] - target[t]); // dL/dlogits = 2 * (pred - target)
    }

    dense_backward(value_head, input, batch_size, dL_dlogits, dL_dw, dL_db, dL_dinput);
}

// Calcul des avantages via la méthode GAE (Generalized Advantage Estimation)
void compute_advantages(double rewards[BATCH_SIZE], double values[BATCH_SIZE], int done[BATCH_SIZE], double advantages[BATCH_SIZE]) {
    advantages[BATCH_SIZE-1] = 0;
    for (int t=BATCH_SIZE-2; t>=0; t--) {
        double delta = rewards[t] + GAMMA * values[t+1] * (1 - done[t]) - values[t];
        advantages[t] = delta + GAMMA * LAMBDA * advantages[t+1] * (1 - done[t]);
    }
}

// Fonction de coût de la policy head/actor i.e. L^CLIP = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] avec r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
double ppo_loss(double prob[BATCH_SIZE][POLICY_OUTPUT_SIZE], double old_prob[BATCH_SIZE][POLICY_OUTPUT_SIZE], double advantages[BATCH_SIZE], int actions[BATCH_SIZE], double dlogp[BATCH_SIZE]) {
    double loss = 0;

    for (int t=0; t<BATCH_SIZE; t++) {
        double unclipped = prob[t][actions[t]] / old_prob[t][actions[t]];

        double clipped = unclipped;
        if (clipped > 1 + EPSILON) {
            clipped = 1 + EPSILON;
            if (advantages[t] > 0) {
                dlogp[t] = 0;
            } else {
                dlogp[t] = -advantages[t] * unclipped; // dL/dlogp = -A_t * r_t
            }
        } else if (clipped < 1 - EPSILON) {
            clipped = 1 - EPSILON;
            if (advantages[t] < 0) {
                dlogp[t] = 0;
            } else {
                dlogp[t] = -advantages[t] * unclipped; // dL/dlogp = -A_t * r_t
            }
        } else {
            dlogp[t] = -advantages[t] * unclipped; // dL/dlogp = -A_t * r_t
        }

        double min = fmin(unclipped * advantages[t], clipped * advantages[t]);
        loss -= min;
    }

    loss /= BATCH_SIZE;
    return loss;
}

// Rétropropagation de la policy head/actor (y compris la fonction softmax)
void policy_backward(Dense* policy_head, Trajectory* trajectories, double new_probs[BATCH_SIZE][POLICY_OUTPUT_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[BATCH_SIZE][HIDDEN_SIZE]) {
    double advantages[BATCH_SIZE];
    compute_advantages(trajectories->rewards, trajectories->values, trajectories->done, advantages);

    double dlogp[BATCH_SIZE];
    double loss = ppo_loss(new_probs, trajectories->probs, advantages, trajectories->actions, dlogp);

    double dL_dlogits[BATCH_SIZE][MAX_OUTPUT_SIZE];
    for (int t=0; t<BATCH_SIZE; t++) {
        for (int i=0; i<POLICY_OUTPUT_SIZE; i++) {
            dL_dlogits[t][i] = dlogp[t] * (new_probs[t][i] - (i == trajectories->actions[t])); // dL/dlogits = dL/dlogp * (dlogp/dlogits) avec dlogp/dlogits = new_probs - one_hot(actions)s
        }
    }

    dense_backward(policy_head, trajectories->hidden_states, BATCH_SIZE, dL_dlogits, dL_dw, dL_db, dL_dinput);
}