#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "lstm.h"

// Distribution normale (moyenne 0, écart-type 1) avec la méthode de Box-Muller
double rand_normal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10; // Pas de log(0)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// Produit scalaire canonique de R^n
double dot_product(double* x, double* y, int n) {
    double res = 0.0;
    for (int i = 0; i < n; i++) {
        res += x[i] * y[i];
    }
    return res;
}

// Norme euclidienne associée au produit scalaire canonique
double norm(double* x, int n) {
    return sqrt(dot_product(x, x, n));
}

// Extraction d'une colonne d'une matrice
double* column(double** matrix, int j, int rows) {
    double* col = malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
        col[i] = matrix[i][j];
    }
    return col;
}

// Initialisation orthogonale : on prend la décomposition QR d'une matrice aléatoire et on utilise Q
void orthogonal_init(double** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand_normal();
        }
    }
    
    // Gram-Schmidt
    for (int j=0; j<cols; j++) {
        // Orthogonalisation
        double* projection = malloc(rows * sizeof(double));
        for (int i=0; i<rows; i++) {
            projection[i] = 0.0;
        }

        double* column_j = column(matrix, j, rows);

        for (int k=0; k<j; k++) {
            double* column_k = column(matrix, k, rows);

            double dot = dot_product(column_j, column_k, rows);
            for (int i=0; i<rows; i++) {
                projection[i] += dot * matrix[i][k];
            }

            free(column_k);
        }

        for (int i=0; i<rows; i++) {
            matrix[i][j] -= projection[i];
        }

        free(projection);
        free(column_j);

        column_j = column(matrix, j, rows);

        // Normalisation
        double n = norm(column_j, rows);
        if (n > 1e-6) { // Pas de division par zéro
            for (int i=0; i<rows; i++) {
                matrix[i][j] /= n;
            }
        }

        free(column_j);
    }
}

// Initialisation de Xavier : distribution uniforme dans [-limit, limit] avec limit = sqrt(6 / (input_size + hidden_size))
void xavier_init(double** matrix, int input_size, int hidden_size) {
    double limit = sqrt(6.0 / (input_size + hidden_size));
    for (int i=0; i<hidden_size; i++) {
        for (int j=0; j<input_size; j++) {
            double r = (double)rand() / RAND_MAX;
            double val = (r * 2.0 - 1.0) * limit;
            matrix[i][j] = val;
        }
    }
}

void init_weights(double** matrix, int input_size, int hidden_size) {
    double** wx = malloc(hidden_size * sizeof(double*));
    double** wh = malloc(hidden_size * sizeof(double*));

    for (int i=0; i<hidden_size; i++) {
        wx[i] = malloc(input_size * sizeof(double));
        wh[i] = malloc(hidden_size * sizeof(double));

        for (int j=0; j<input_size; j++) {
            wx[i][j] = matrix[i][j];
        }
        for (int j=0; j<hidden_size; j++) {
            wh[i][j] = matrix[i][input_size + j];
        }
    }

    xavier_init(wx, hidden_size, hidden_size);
    orthogonal_init(wh, hidden_size, input_size);

    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            matrix[i][j] = wx[i][j];
        }
        for (int j = 0; j < hidden_size; j++) {
            matrix[i][input_size + j] = wh[i][j];
        }
        
        free(wx[i]);
        free(wh[i]);
    }

    free(wx);
    free(wh);
}

Lstm* init_lstm(int input_size, int hidden_size, int output_size) {
    Lstm* lstm = malloc(sizeof(Lstm));

    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;
    lstm->output_size = output_size;

    lstm->hidden_state = calloc(hidden_size, sizeof(double));
    lstm->cell_state = calloc(hidden_size, sizeof(double));

    lstm->wf = malloc(hidden_size * sizeof(double*));
    lstm->wi = malloc(hidden_size * sizeof(double*));
    lstm->wc = malloc(hidden_size * sizeof(double*));
    lstm->wo = malloc(hidden_size * sizeof(double*));

    lstm->wf_m = malloc(hidden_size * sizeof(double*));
    lstm->wi_m = malloc(hidden_size * sizeof(double*));
    lstm->wc_m = malloc(hidden_size * sizeof(double*));
    lstm->wo_m = malloc(hidden_size * sizeof(double*));

    lstm->wf_v = malloc(hidden_size * sizeof(double*));
    lstm->wi_v = malloc(hidden_size * sizeof(double*));
    lstm->wc_v = malloc(hidden_size * sizeof(double*));
    lstm->wo_v = malloc(hidden_size * sizeof(double*));

    int col_size = input_size + hidden_size;
    
    for (int i=0; i<hidden_size; i++) {
        lstm->wf[i] = calloc(col_size, sizeof(double));
        lstm->wi[i] = calloc(col_size, sizeof(double));
        lstm->wc[i] = calloc(col_size, sizeof(double));
        lstm->wo[i] = calloc(col_size, sizeof(double));

        lstm->wf_m[i] = calloc(col_size, sizeof(double));
        lstm->wi_m[i] = calloc(col_size, sizeof(double));
        lstm->wc_m[i] = calloc(col_size, sizeof(double));
        lstm->wo_m[i] = calloc(col_size, sizeof(double));

        lstm->wf_v[i] = calloc(col_size, sizeof(double));
        lstm->wi_v[i] = calloc(col_size, sizeof(double));
        lstm->wc_v[i] = calloc(col_size, sizeof(double));
        lstm->wo_v[i] = calloc(col_size, sizeof(double));
    }

    init_weights(lstm->wf, input_size, hidden_size);
    init_weights(lstm->wi, input_size, hidden_size);
    init_weights(lstm->wc, input_size, hidden_size);
    init_weights(lstm->wo, input_size, hidden_size);

    lstm->bf = calloc(hidden_size, sizeof(double));
    lstm->bi = calloc(hidden_size, sizeof(double));
    lstm->bc = calloc(hidden_size, sizeof(double));
    lstm->bo = calloc(hidden_size, sizeof(double));

    lstm->bf_m = calloc(hidden_size, sizeof(double));
    lstm->bi_m = calloc(hidden_size, sizeof(double));
    lstm->bc_m = calloc(hidden_size, sizeof(double));
    lstm->bo_m = calloc(hidden_size, sizeof(double));

    lstm->bf_v = calloc(hidden_size, sizeof(double));
    lstm->bi_v = calloc(hidden_size, sizeof(double));
    lstm->bc_v = calloc(hidden_size, sizeof(double));
    lstm->bo_v = calloc(hidden_size, sizeof(double));
    
    return lstm;
}