#ifndef POLICY_H
#define POLICY_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "struct.h"
#include "func.h"
#include "reward.h"

double* forward(LSTM* network, double* data, double temperature);
double* backpropagation(LSTM* network, double* data, double learningRate, int steps, trajectory** trajectories, int batchCount, double temperature, double epsilon);
double* discountedPNL(double* rewards, double gamma, int steps, bool normalize);

#endif