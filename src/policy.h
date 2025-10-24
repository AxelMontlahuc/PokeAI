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

typedef struct {
	double grad_norm;
	double clip_scale;
} BackpropStats;

double* forward(LSTM* network, double* data, double temperature);
void backpropagation(
	LSTM* network,
	double learningRate,
	int steps,
	trajectory** trajectories,
	int batchCount,
	double temperature,
	double epsilon,
	BackpropStats* stats
);

#endif