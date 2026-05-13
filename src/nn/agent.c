#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "agent.h"
#include "ppo.h"
#include "lstm.h"
#include "adam.h"
#include "config.h"
#include "../game/reward.h"

// Initialisation de l'agent
Agent* init_agent() {
    Agent* agent = malloc(sizeof(Agent));

    init_lstm(&agent->lstm, INPUT_SIZE, HIDDEN_SIZE);
    init_dense(&agent->policy_head, HIDDEN_SIZE, POLICY_OUTPUT_SIZE);
    init_dense(&agent->value_head, HIDDEN_SIZE, VALUE_OUTPUT_SIZE);

    return agent;
}

// Fonction d'activation softmax (softmax = e^x / Σ(e^x)) pour convertir les "logits" (sorties brutes de la politique) en une distribution de probabilité
void softmax(double* logits, int size, double* output) {
    double max_logit = logits[0];
    for (int i=1; i<size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum = 0;
    for (int i=0; i<size; i++) {
        sum += exp(logits[i] - max_logit); // On soustrait le maximum pour des questions de stabilité numérique
    }
    for (int i=0; i<size; i++) {
        output[i] = exp(logits[i] - max_logit) / sum;
    }
}

int action_choice(double probs[POLICY_OUTPUT_SIZE]) {
    double r = (double)rand() / RAND_MAX;
    double s = 0;
    for (int i=0; i<POLICY_OUTPUT_SIZE; i++) {
        s += probs[i];
        if (r < s) {
            return i;
        }
    }
    return 0;
}

// Propagation pour tout l'agent
void agent_forward_t(Agent* agent, double state[INPUT_SIZE], Trajectory* traj, int t) {
    lstm_forward(&agent->lstm, traj, state, t);

    dense_forward(&agent->policy_head, agent->lstm.hidden_state, agent->policy_logits);
    dense_forward(&agent->value_head, agent->lstm.hidden_state, agent->value_logits);

    softmax(agent->policy_logits, agent->policy_head.output_size, traj->probs[t]);

    traj->actions[t] = action_choice(traj->probs[t]);
    traj->values[t] = agent->value_logits[0];

    if (is_done(state)) {
        traj->done[t] = 1; // Requis pour la GAE
    } else {
        traj->done[t] = 0;
    }

    if (t == 0) {
        traj->rewards[t] = 0;
    } else {
        traj->rewards[t] = reward(traj->states[t-1], state);    
    }

    memcpy(traj->states[t], state, sizeof(double) * INPUT_SIZE);
}

void agent_forward(Agent* agent, Trajectory* traj) {
    for (int t=0; t<BATCH_SIZE; t++) {
        double state[INPUT_SIZE]; // TODO : Récupérer l'état depuis l'émulateur
        agent_forward_t(agent, state, traj, t);
        // TODO : Envoyer l'action choisie à l'émulateur
    }
}

// Rétropropagation pour tout l'agent
void recompute_probs(Agent* agent, Trajectory* old_traj, double new_probs[BATCH_SIZE][POLICY_OUTPUT_SIZE]) {
    for (int t=0; t<BATCH_SIZE; t++) {
        dense_forward(&agent->policy_head, old_traj->hidden_states[t], agent->policy_logits);
        softmax(agent->policy_logits, agent->policy_head.output_size, new_probs[t]);
    }
}

void agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj) {
    // Rétropropagation de la value head/critic
    double returns[BATCH_SIZE];
    compute_advantages(traj->rewards, traj->values, traj->done, returns);
    for (int t=0; t<BATCH_SIZE; t++) {
        returns[t] += traj->values[t]; // R_t = A_t + V_t (les returns correspondent aux récompenses sur le long terme, c'est-à-dire la somme pondérées des récompenses futures)
    }

    double dL_dw_v[MAX_OUTPUT_SIZE][HIDDEN_SIZE] = {0};
    double dL_db_v[MAX_OUTPUT_SIZE] = {0};
    double dL_dinput_v[BATCH_SIZE][HIDDEN_SIZE] = {0};
    value_backward(&agent->value_head, traj->values, returns, traj->hidden_states, dL_dw_v, dL_db_v, dL_dinput_v);

    // Rétropropagation de la policy head/actor
    double new_probs[BATCH_SIZE][POLICY_OUTPUT_SIZE];
    recompute_probs(agent, traj, new_probs);

    double dL_dw_p[MAX_OUTPUT_SIZE][HIDDEN_SIZE] = {0};
    double dL_db_p[MAX_OUTPUT_SIZE] = {0};
    double dL_dinput_p[BATCH_SIZE][HIDDEN_SIZE] = {0};
    policy_backward(&agent->policy_head, traj, new_probs, dL_dw_p, dL_db_p, dL_dinput_p);

    // Mise à jour des poids avec Adam
    optim->t += 1;

    optimizer_step_matrix_1(optim, agent->value_head.w, agent->value_head.w_m, agent->value_head.w_v, dL_dw_v, agent->value_head.output_size, agent->value_head.input_size);
    optimizer_step_vector_1(optim, agent->value_head.b, agent->value_head.b_m, agent->value_head.b_v, dL_db_v, agent->value_head.output_size);

    optimizer_step_matrix_1(optim, agent->policy_head.w, agent->policy_head.w_m, agent->policy_head.w_v, dL_dw_p, agent->policy_head.output_size, agent->policy_head.input_size);
    optimizer_step_vector_1(optim, agent->policy_head.b, agent->policy_head.b_m, agent->policy_head.b_v, dL_db_p, agent->policy_head.output_size);

    // Rétropropagation à travers le LSTM
    double dL_dwf[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwi[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwc[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dwo[HIDDEN_SIZE][COL_SIZE] = {0};
    double dL_dbf[HIDDEN_SIZE] = {0};
    double dL_dbi[HIDDEN_SIZE] = {0};
    double dL_dbc[HIDDEN_SIZE] = {0};
    double dL_dbo[HIDDEN_SIZE] = {0};

    double c_ini[HIDDEN_SIZE] = {0}; // On suppose que la mémoire à long terme est initialisée à zéro au début de chaque trajectoire pour l'instant, mais il faudra à terme le stocker et le faire passer d'une trajectoire à l'autre
    lstm_backward(&agent->lstm, traj, dL_dinput_v, dL_dinput_p, c_ini, dL_dwf, dL_dwi, dL_dwc, dL_dwo, dL_dbf, dL_dbi, dL_dbc, dL_dbo);

    // Mise à jour des poids du LSTM avec Adam
    optimizer_step_matrix_2(optim, agent->lstm.wf, agent->lstm.wf_m, agent->lstm.wf_v, dL_dwf, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_matrix_2(optim, agent->lstm.wi, agent->lstm.wi_m, agent->lstm.wi_v, dL_dwi, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_matrix_2(optim, agent->lstm.wc, agent->lstm.wc_m, agent->lstm.wc_v, dL_dwc, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_matrix_2(optim, agent->lstm.wo, agent->lstm.wo_m, agent->lstm.wo_v, dL_dwo, HIDDEN_SIZE, COL_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bf, agent->lstm.bf_m, agent->lstm.bf_v, dL_dbf, HIDDEN_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bi, agent->lstm.bi_m, agent->lstm.bi_v, dL_dbi, HIDDEN_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bc, agent->lstm.bc_m, agent->lstm.bc_v, dL_dbc, HIDDEN_SIZE);
    optimizer_step_vector_2(optim, agent->lstm.bo, agent->lstm.bo_m, agent->lstm.bo_v, dL_dbo, HIDDEN_SIZE);
}

void train_epoch(Agent* agent, Optimizer* optim) {
    Trajectory* traj = malloc(sizeof(Trajectory));

    agent_forward(agent, traj);
    agent_backward(agent, optim, traj);

    free(traj);
}

void train(Agent* agent, Optimizer* optim, int epochs) {
    for (int epoch=0; epoch<epochs; epoch++) {
        train_epoch(agent, optim);
    }
}