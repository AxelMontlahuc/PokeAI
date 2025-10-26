#ifndef REWARD_H
#define REWARD_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "state.h"
#include "constants.h"

double pnl(state s, state s_next);
void normPNL(double* G, int n);
void computeGAE(
	double* rewards,
	double* values,
	int steps,
	double gamma,
	double lambda,
	double* out_advantages,
	double* out_returns
);
void resetFlags();


#endif