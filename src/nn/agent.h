#ifndef AGENT_H
#define AGENT_H

#include "lstm.h"
#include "dense.h"
#include "adam.h"
#include "config.h"

typedef struct Agent Agent;

typedef struct EpochSummary {
    float ppo_loss_sum;
    float value_loss_sum;
    float reward_sum;
    float value_sum;
    float value_sq_sum;
    float adv_sum;
    float adv_sq_sum;
    float ratio_sum;
    float unclipped_ratio_sum;
    float clipped_count;
    float entropy_sum;
    float kl_sum;
    float action_counts[POLICY_OUTPUT_SIZE];
    int minibatch_count;
} EpochSummary;

struct Agent {
    Lstm lstm;
    Dense policy_head;
    Dense value_head;
    
    float policy_logits[POLICY_OUTPUT_SIZE];
    float value_logits[VALUE_OUTPUT_SIZE];
    
    float entropy_coeff;
    float temperature;
};

Agent* init_agent();

int action_choice(float probs[POLICY_OUTPUT_SIZE]);
void agent_forward_t(Agent* agent, int state[INPUT_SIZE], Trajectory* traj, int t);
void agent_forward(Agent* agent, Trajectory* traj[NUM_ENVS]);
float agent_backward_minibatch(Agent* agent, Optimizer* optim, Minibatch* minibatch);
float agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj[NUM_ENVS], EpochSummary* summary);
void recompute_probs(Agent* agent, Minibatch* current_minibatch, float new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE]);
void recompute_values(Agent* agent, Minibatch* current_minibatch, float new_values[MINIBATCH_SIZE]);
void train_epoch(Agent* agent, Optimizer* optim);
void train(Agent* agent, Optimizer* optim, int start_epoch, int epochs);

#endif