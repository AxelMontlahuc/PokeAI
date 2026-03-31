#ifndef AGENT_H
#define AGENT_H

#include "lstm.h"
#include "dense.h"
#include "config.h"

typedef struct Agent Agent;

struct Agent {
    Lstm lstm;
    Dense policy_head;
    Dense value_head;
    
    double policy_logits[POLICY_OUTPUT_SIZE];
    double value_logits[VALUE_OUTPUT_SIZE];
    double policy_output[POLICY_OUTPUT_SIZE];
};

Agent* init_agent();
double* agent_forward(Agent* agent, double* input, double** value_output, double** policy_output);

#endif