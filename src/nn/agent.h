#ifndef AGENT_H
#define AGENT_H

#include "lstm.h"
#include "dense.h"
#include "adam.h"
#include "config.h"

typedef struct Agent Agent;

struct Agent {
    Lstm lstm;
    Dense policy_head;
    Dense value_head;
    
    double policy_logits[POLICY_OUTPUT_SIZE];
    double value_logits[VALUE_OUTPUT_SIZE];
};

Agent* init_agent();

int action_choice(double probs[POLICY_OUTPUT_SIZE]);
void agent_forward_t(Agent* agent, int state[INPUT_SIZE], Trajectory* traj, int t);
void agent_forward(Agent* agent, Trajectory* traj[NUM_ENVS]);
void agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj);
void recompute_probs(Agent* agent, Trajectory* old_traj, double new_probs[BATCH_SIZE][POLICY_OUTPUT_SIZE]);
void train_epoch(Agent* agent, Optimizer* optim);
void train(Agent* agent, Optimizer* optim, int epochs);

#endif