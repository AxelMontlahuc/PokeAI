#ifndef PPO_H
#define PPO_H

double** value_backward(Dense* value_head, double** pred, double** target, double** input, int batch_size, double** dL_dw, double* dL_db);

#endif