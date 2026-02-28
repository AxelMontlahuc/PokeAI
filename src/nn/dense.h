#ifndef DENSE_H
#define DENSE_H

typedef struct Dense Dense;

struct Dense {
    int input_size;
    int output_size;

    double** w;
    double** w_m;
    double** w_v;

    double* b;
    double* b_m;
    double* b_v;
};

#endif