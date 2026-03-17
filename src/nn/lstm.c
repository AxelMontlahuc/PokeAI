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

// Fonction auxiliaire pour l'initialisation des poids
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

    xavier_init(wx, input_size, hidden_size);
    orthogonal_init(wh, hidden_size, hidden_size);

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

// Initialisation du lstm
Lstm* init_lstm(int input_size, int hidden_size) {
    Lstm* lstm = malloc(sizeof(Lstm));

    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;

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

// Libération de la mémoire du lstm
void free_lstm(Lstm* lstm) {
    free(lstm->hidden_state);
    free(lstm->cell_state);

    for (int i=0; i<lstm->hidden_size; i++) {
        free(lstm->wf[i]);
        free(lstm->wi[i]);
        free(lstm->wc[i]);
        free(lstm->wo[i]);

        free(lstm->wf_m[i]);
        free(lstm->wi_m[i]);
        free(lstm->wc_m[i]);
        free(lstm->wo_m[i]);

        free(lstm->wf_v[i]);
        free(lstm->wi_v[i]);
        free(lstm->wc_v[i]);
        free(lstm->wo_v[i]);
    }

    free(lstm->wf);
    free(lstm->wi);
    free(lstm->wc);
    free(lstm->wo);

    free(lstm->wf_m);
    free(lstm->wi_m);
    free(lstm->wc_m);
    free(lstm->wo_m);

    free(lstm->wf_v);
    free(lstm->wi_v);
    free(lstm->wc_v);
    free(lstm->wo_v);

    free(lstm->bf);
    free(lstm->bi);
    free(lstm->bc);
    free(lstm->bo);

    free(lstm->bf_m);
    free(lstm->bi_m);
    free(lstm->bc_m);
    free(lstm->bo_m);

    free(lstm->bf_v);
    free(lstm->bi_v);
    free(lstm->bc_v);
    free(lstm->bo_v);

    free(lstm);
}

// Fonction sigmoïde
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double* matrix_vector_product(double** matrix, double* vector, int rows, int cols) {
    double* result = malloc(rows * sizeof(double));
    for (int i=0; i<rows; i++) {
        result[i] = 0.0;
        for (int j=0; j<cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

// Propagation
void lstm_forward(Lstm* lstm, double* input) {
    // Concaténation de l'entrée et de l'état caché dans un array z
    int z_size = lstm->input_size + lstm->hidden_size;
    double* z = malloc(z_size * sizeof(double));
    for (int i=0; i<lstm->input_size; i++) {
        z[i] = input[i];
    }
    for (int i=0; i<lstm->hidden_size; i++) {
        z[lstm->input_size + i] = lstm->hidden_state[i];
    }

    // Calcul des portes
    double* f = malloc(lstm->hidden_size * sizeof(double));
    double* i = malloc(lstm->hidden_size * sizeof(double));
    double* g = malloc(lstm->hidden_size * sizeof(double)); // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    double* o = malloc(lstm->hidden_size * sizeof(double));

    double* wfz = matrix_vector_product(lstm->wf, z, lstm->hidden_size, z_size);
    double* wiz = matrix_vector_product(lstm->wi, z, lstm->hidden_size, z_size);
    double* wcz = matrix_vector_product(lstm->wc, z, lstm->hidden_size, z_size);
    double* woz = matrix_vector_product(lstm->wo, z, lstm->hidden_size, z_size);

    for (int j=0; j<lstm->hidden_size; j++) {
        f[j] = sigmoid(wfz[j] + lstm->bf[j]);
        i[j] = sigmoid(wiz[j] + lstm->bi[j]);
        g[j] = tanh(wcz[j] + lstm->bc[j]);
        o[j] = sigmoid(woz[j] + lstm->bo[j]);
    }

    free(wfz);
    free(wiz);
    free(wcz);
    free(woz);

    // Mise à jour de la mémoire long-terme (cellule)
    for (int j=0; j<lstm->hidden_size; j++) {
        lstm->cell_state[j] = f[j] * lstm->cell_state[j] + i[j] * g[j];
    }

    // Mise à jour de la mémoire court-terme (état caché)
    for (int j=0; j<lstm->hidden_size; j++) {
        lstm->hidden_state[j] = o[j] * tanh(lstm->cell_state[j]);
    }

    free(f);
    free(i);
    free(g);
    free(o);
    free(z);
}