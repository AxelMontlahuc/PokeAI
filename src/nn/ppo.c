#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ppo.h"
#include "dense.h"
#include "config.h"
#include "agent.h"

// Fonction coût de la value head/critic (MSE)
double value_loss(double pred[MINIBATCH_SIZE], double target[MINIBATCH_SIZE]) {
    double loss = 0;

    for (int i=0; i<MINIBATCH_SIZE; i++) {
        double diff = pred[i] - target[i];
        loss += diff * diff;
    }
    loss /= MINIBATCH_SIZE;

    return loss;
}

// Rétropropagation de la value head/critic
double value_backward(Dense* value_head, double pred[MINIBATCH_SIZE], double target[MINIBATCH_SIZE], double input[MINIBATCH_SIZE][HIDDEN_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE]) {
    double dL_dlogits[MINIBATCH_SIZE][MAX_OUTPUT_SIZE];

    for (int t=0; t<MINIBATCH_SIZE; t++) {
        dL_dlogits[t][0] = 2 * (pred[t] - target[t]); // dL/dlogits = 2 * (pred - target)
    }

    dense_backward(value_head, input, dL_dlogits, dL_dw, dL_db, dL_dinput);
    return value_loss(pred, target);
}

// Calcul des avantages via la méthode GAE (Generalized Advantage Estimation)
void compute_advantages(double rewards[TRAJ_SIZE], double values[TRAJ_SIZE], int done[TRAJ_SIZE], double advantages[TRAJ_SIZE]) {
    advantages[TRAJ_SIZE-1] = rewards[TRAJ_SIZE-1] - values[TRAJ_SIZE-1];
    for (int t=TRAJ_SIZE-2; t>=0; t--) {
        double delta = rewards[t] + GAMMA * values[t+1] * (1 - done[t]) - values[t];
        advantages[t] = delta + GAMMA * LAMBDA * advantages[t+1] * (1 - done[t]);
    }
}

// Fonction de coût de la policy head/actor i.e. L^CLIP = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ] avec r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
double ppo_loss(Minibatch* minibatch, double prob[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], double old_prob[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], double advantages[MINIBATCH_SIZE], int actions[MINIBATCH_SIZE], double dlogp[MINIBATCH_SIZE], double ent_coeff) {
    double loss = 0;

    for (int t=0; t<MINIBATCH_SIZE; t++) {
        double unclipped = prob[t][actions[t]] / (old_prob[t][actions[t]] + 1e-10); // On évite de diviser par zéro

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

        minibatch->ratios[t] = fmin(unclipped, clipped);
        minibatch->unclipped_ratios[t] = unclipped;
        minibatch->clipped[t] = (clipped != unclipped);
        double min = fmin(unclipped * advantages[t], clipped * advantages[t]);
        loss -= min;
    }

    // Calcul de l'entropie
    double mean_entropy = 0;
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        double current_entropy = 0;
        for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
            current_entropy -= prob[t][j] * log(prob[t][j] + 1e-10); // On évite log(0)
        }
        mean_entropy += current_entropy;
    }
    mean_entropy /= MINIBATCH_SIZE;

    loss /= MINIBATCH_SIZE;
    loss -= ent_coeff * mean_entropy;  // Encourage l'exploration
    return loss;
}

// Rétropropagation de la policy head/actor (y compris la fonction softmax)
void policy_backward(Dense* policy_head, Minibatch* minibatch, double new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE], double dL_dw[MAX_OUTPUT_SIZE][HIDDEN_SIZE], double dL_db[MAX_OUTPUT_SIZE], double dL_dinput[MINIBATCH_SIZE][HIDDEN_SIZE], double ent_coeff) {
    double dlogp[MINIBATCH_SIZE];

    // On normalise (moyenne à zéro et écart-type de un) et clip (à 5.0) les avantages (pour plus de stabilité)
    double norm_adv[MINIBATCH_SIZE];
    double mean = 0.0;
    for (int t = 0; t < MINIBATCH_SIZE; t++) {
        mean += minibatch->advantages[t];
    }
    mean /= MINIBATCH_SIZE;

    double var = 0.0;
    for (int t = 0; t < MINIBATCH_SIZE; t++) {
        double diff = minibatch->advantages[t] - mean;
        var += diff * diff;
    }
    var /= MINIBATCH_SIZE;
    double std = sqrt(var + 1e-8);

    const double ADV_CLIP = 5.0;
    for (int t = 0; t < MINIBATCH_SIZE; t++) {
        norm_adv[t] = (minibatch->advantages[t] - mean) / std;
        
        if (norm_adv[t] > ADV_CLIP) {
            norm_adv[t] = ADV_CLIP;
        }
        if (norm_adv[t] < -ADV_CLIP) {
            norm_adv[t] = -ADV_CLIP;
        }
    }

    minibatch->ppo_loss = ppo_loss(minibatch, new_probs, minibatch->probs, norm_adv, minibatch->actions, dlogp, ent_coeff);

    double dL_dlogits[MINIBATCH_SIZE][MAX_OUTPUT_SIZE];
    for (int t=0; t<MINIBATCH_SIZE; t++) {
        double entropy = 0;
        for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
            entropy -= new_probs[t][j] * log(new_probs[t][j] + 1e-10);
        }

        for (int i=0; i<POLICY_OUTPUT_SIZE; i++) {
            dL_dlogits[t][i] = dlogp[t] * (new_probs[t][i] - (i == minibatch->actions[t])); // dL/dlogits = dL/dlogp * (dlogp/dlogits) + dL/dlogits_h avec dlogp/dlogits = new_probs - one_hot(actions)s
            dL_dlogits[t][i] += ent_coeff * new_probs[t][i] * (log(new_probs[t][i] + 1e-10) + entropy); // dL/dlogits_h = -ENTROPY_COEFF * (log(new_probs) + entropy) * new_probs
        }

        minibatch->kl[t] = 0;
        for (int j=0; j<POLICY_OUTPUT_SIZE; j++) {
            double p = minibatch->probs[t][j];
            double q = new_probs[t][j];
            minibatch->kl[t] += p * (log(p + 1e-10) - log(q + 1e-10)); // KL(π_old || π_new) = Σ(π_old * (log(π_old) - log(π_new)))
        }
    }

    dense_backward(policy_head, minibatch->hidden_states, dL_dlogits, dL_dw, dL_db, dL_dinput);
}