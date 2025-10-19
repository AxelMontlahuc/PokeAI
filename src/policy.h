#ifndef POLICY_H
#define POLICY_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "struct.h"
#include "func.h"

double* forward(LSTM* network, double* data);
double* backpropagation(LSTM* network, double* data, double learningRate, int steps, trajectory* trajectories);
double pnl(state s_t, state s_t_suite);
double* discountedPNL(state* etats, double gamma, int nb_traj);
double* softmaxLayer(double* logits, int n);

#endif