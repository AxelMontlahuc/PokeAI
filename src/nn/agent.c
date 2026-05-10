#include <stdlib.h>
#include <math.h>

#include "agent.h"
#include "config.h"

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

// Propagation pour tout l'agent
double* agent_forward(Agent* agent, double* input, double** value_output, double** policy_output) {
    lstm_forward(&agent->lstm, input);

    dense_forward(&agent->policy_head, agent->lstm.hidden_state, agent->policy_logits);
    dense_forward(&agent->value_head, agent->lstm.hidden_state, agent->value_logits);
    
    *policy_output = agent->policy_logits;
    *value_output = agent->value_logits;

    softmax(agent->policy_logits, agent->policy_head.output_size, agent->policy_output);
    return agent->policy_output;
}

// Rétropropagation pour tout l'agent
void agent_backward(Agent* agent) {
    // Rétropropagation de la value head/critic
    // value_backward(&agent->value_head, ...);
}