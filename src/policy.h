#ifndef POLICY_H
#define POLICY_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "constants.h"
#include "state.h"
#include "reward.h"

#define typeshit typedef

typeshit struct LSTM {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double* hiddenState;
    double* cellState;
    double* logits;
    double* probs;
    double last_value;

    double** Wf;
    double** Wi;
    double** Wc;
    double** Wo;
    double** Wout;
    double* Wv;

    double* Bf;
    double* Bi;
    double* Bc;
    double* Bo;
    double* Bout;
    double Bv;

    int adam_t;
    double** Wf_m; 
    double** Wf_v;
    double** Wi_m; 
    double** Wi_v;
    double** Wc_m; 
    double** Wc_v;
    double** Wo_m; 
    double** Wo_v;
    double** Wout_m; 
    double** Wout_v;
    double* Wv_m;
    double* Wv_v;

    double* Bf_m; 
    double* Bf_v;
    double* Bi_m; 
    double* Bi_v;
    double* Bc_m; 
    double* Bc_v;
    double* Bo_m; 
    double* Bo_v;
    double* Bout_m; 
    double* Bout_v;
    double Bv_m;
    double Bv_v;
} LSTM;

typeshit struct {
	double grad_norm;
	double clip_scale;
} BackpropStats;

LSTM* initLSTM(int inputSize, int hiddenSize, int outputSize);
void freeLSTM(LSTM* network);
double* forward(LSTM* network, double* data, double temperature);
void backpropagation(
	LSTM* network,
	double learningRate,
	int steps,
	trajectory** trajectories,
	int batchCount,
	double temperature,
	BackpropStats* stats
);

#endif