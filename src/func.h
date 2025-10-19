#ifndef FUNC_H
#define FUNC_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

double sigmoid(double x);
double mse(double pred, double target);
double softmax(double* tab, int n, int i);

#endif