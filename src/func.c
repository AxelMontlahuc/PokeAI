#include "func.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double mse(double pred, double target) {
    return (pred - target) * (pred - target);
}

double softmax(double* tab, int n, int i) {
    double sum = 0.0;
    for (int j=0; j<n; j++) {
        sum += exp(tab[j]);
    }
    return exp(tab[i]) / sum;
}

