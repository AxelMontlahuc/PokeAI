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

Dense* init_dense(int input_size, int output_size);
void free_dense(Dense* dense);
double* dense_forward(Dense* dense, double* input);
double** dense_backward(Dense* dense, double** input, int batch_size, double** dL_dlogits, double** dL_dw, double* dL_db);

#endif