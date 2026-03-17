#ifndef AGENT_H
#define AGENT_H

#include "lstm.h"
#include "dense.h"

typedef struct Agent Agent;

struct Agent {
    Lstm* lstm;
    Dense* policy_head;
    Dense* value_head;
};

Agent* init_agent();
void free_agent(Agent* agent);
double* agent_forward(Agent* agent, double* input, double** value_output, double** policy_output);

#endif