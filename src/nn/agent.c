#include <stdlib.h>
#include <math.h>

#include "agent.h"
#include "config.h"

// Initialisation de l'agent
Agent* init_agent() {
    Agent* agent = malloc(sizeof(Agent));

    agent->lstm = init_lstm(INPUT_SIZE, HIDDEN_SIZE);
    agent->policy_head = init_dense(HIDDEN_SIZE, POLICY_OUTPUT_SIZE);
    agent->value_head = init_dense(HIDDEN_SIZE, VALUE_OUTPUT_SIZE);

    return agent;
}

// Libération de la mémoire de l'agent
void free_agent(Agent* agent) {
    free_lstm(agent->lstm);
    free_dense(agent->policy_head);
    free_dense(agent->value_head);
    free(agent);
}

// Fonction d'activation softmax (softmax = e^x / Σ(e^x)) pour convertir les "logits" (sorties brutes de la politique) en une distribution de probabilité
double* softmax(double* logits, int size) {
    double* output = malloc(size * sizeof(double));

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

    return output;
}

// Propagation pour tout l'agent
double* agent_forward(Agent* agent, double* input, double** value_output, double** policy_output) {
    lstm_forward(agent->lstm, input);

    *policy_output = dense_forward(agent->policy_head, agent->lstm->hidden_state);
    *value_output = dense_forward(agent->value_head, agent->lstm->hidden_state);

    double* output = softmax(*policy_output, agent->policy_head->output_size);
    return output;
}