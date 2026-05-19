#ifndef AGENT_H
#define AGENT_H

#include "lstm.h"
#include "dense.h"
#include "adam.h"
#include "config.h"

typedef struct Agent Agent;

typedef struct EpochSummary {
    double ppo_loss_sum;
    double value_loss_sum;
    double reward_sum;
    double value_sum;
    double value_sq_sum;
    double adv_sum;
    double adv_sq_sum;
    double ratio_sum;
    double unclipped_ratio_sum;
    double clipped_count;
    double entropy_sum;
    double kl_sum;
    double action_counts[POLICY_OUTPUT_SIZE];
    int minibatch_count;
} EpochSummary;

struct Agent {
    Lstm lstm;
    Dense policy_head;
    Dense value_head;
    
    double policy_logits[POLICY_OUTPUT_SIZE];
    double value_logits[VALUE_OUTPUT_SIZE];
    
    double entropy_coeff;
    double temperature;
};

Agent* init_agent();

int action_choice(double probs[POLICY_OUTPUT_SIZE]);
void agent_forward_t(Agent* agent, int state[INPUT_SIZE], Trajectory* traj, int t);
void agent_forward(Agent* agent, Trajectory* traj[NUM_ENVS]);
double agent_backward_minibatch(Agent* agent, Optimizer* optim, Minibatch* minibatch);
double agent_backward(Agent* agent, Optimizer* optim, Trajectory* traj[NUM_ENVS], EpochSummary* summary);
void recompute_probs(Agent* agent, Minibatch* current_minibatch, double new_probs[MINIBATCH_SIZE][POLICY_OUTPUT_SIZE]);
void train_epoch(Agent* agent, Optimizer* optim);
void train(Agent* agent, Optimizer* optim, int start_epoch, int epochs);
double get_entropy_coeff();
double get_temperature();
void set_epoch_schedule(int epoch);

#endif