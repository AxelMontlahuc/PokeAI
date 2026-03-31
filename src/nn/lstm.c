#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "lstm.h"
#include "config.h"

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
void column(double** matrix, int j, int rows, double* col) {
    for (int i = 0; i < rows; i++) {
        col[i] = matrix[i][j];
    }
}

// Initialisation orthogonale : on prend la décomposition QR d'une matrice aléatoire et on utilise Q
void orthogonal_init(double** matrix, int rows, int cols) {
    double projection[MAX_HIDDEN_SIZE];
    double column_j_buf[MAX_HIDDEN_SIZE];
    double column_k_buf[MAX_HIDDEN_SIZE];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand_normal();
        }
    }
    
    // Gram-Schmidt
    for (int j=0; j<cols; j++) {
        // Orthogonalisation
        for (int i=0; i<rows; i++) {
            projection[i] = 0.0;
        }

        column(matrix, j, rows, column_j_buf);

        for (int k=0; k<j; k++) {
            column(matrix, k, rows, column_k_buf);

            double dot = dot_product(column_j_buf, column_k_buf, rows);
            for (int i=0; i<rows; i++) {
                projection[i] += dot * matrix[i][k];
            }
        }

        for (int i=0; i<rows; i++) {
            matrix[i][j] -= projection[i];
        }

        column(matrix, j, rows, column_j_buf);

        // Normalisation
        double n = norm(column_j_buf, rows);
        if (n > 1e-6) { // Pas de division par zéro
            for (int i=0; i<rows; i++) {
                matrix[i][j] /= n;
            }
        }
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
    double wx[MAX_HIDDEN_SIZE][MAX_INPUT_SIZE];
    double wh[MAX_HIDDEN_SIZE][MAX_HIDDEN_SIZE];

    for (int i=0; i<hidden_size; i++) {
        for (int j=0; j<input_size; j++) {
            wx[i][j] = matrix[i][j];
        }
        for (int j=0; j<hidden_size; j++) {
            wh[i][j] = matrix[i][input_size + j];
        }
    }

    xavier_init((double**)wx, input_size, hidden_size);
    orthogonal_init((double**)wh, hidden_size, hidden_size);

    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            matrix[i][j] = wx[i][j];
        }
        for (int j = 0; j < hidden_size; j++) {
            matrix[i][input_size + j] = wh[i][j];
        }
    }
}

// Initialisation du lstm
void init_lstm(Lstm* lstm, int input_size, int hidden_size) {
    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;
    
    for (int i = 0; i < input_size; i++) {
        lstm->hidden_state[i] = 0.0;
        lstm->cell_state[i] = 0.0;
    }

    for (int i = 0; i < hidden_size; i++) {
        lstm->bf[i] = 0.0;
        lstm->bi[i] = 0.0;
        lstm->bc[i] = 0.0;
        lstm->bo[i] = 0.0;

        lstm->bf_m[i] = 0.0;
        lstm->bi_m[i] = 0.0;
        lstm->bc_m[i] = 0.0;
        lstm->bo_m[i] = 0.0;

        lstm->bf_v[i] = 0.0;
        lstm->bi_v[i] = 0.0;
        lstm->bc_v[i] = 0.0;
        lstm->bo_v[i] = 0.0;
    }

    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size + hidden_size; j++) {
            lstm->wf[i][j] = 0.0;
            lstm->wi[i][j] = 0.0;
            lstm->wc[i][j] = 0.0;
            lstm->wo[i][j] = 0.0;

            lstm->wf_m[i][j] = 0.0;
            lstm->wi_m[i][j] = 0.0;
            lstm->wc_m[i][j] = 0.0;
            lstm->wo_m[i][j] = 0.0;

            lstm->wf_v[i][j] = 0.0;
            lstm->wi_v[i][j] = 0.0;
            lstm->wc_v[i][j] = 0.0;
            lstm->wo_v[i][j] = 0.0;
        }
    }

    init_weights(lstm->wf, input_size, hidden_size);
    init_weights(lstm->wi, input_size, hidden_size);
    init_weights(lstm->wc, input_size, hidden_size);
    init_weights(lstm->wo, input_size, hidden_size);
}

// Fonction sigmoïde
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double matrix_vector_product(double** matrix, double* vector, int rows, int cols, double* result) {
    for (int i=0; i<rows; i++) {
        result[i] = 0.0;
        for (int j=0; j<cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Propagation
void lstm_forward(Lstm* lstm, double* input) {
    // Concaténation de l'entrée et de l'état caché dans un array z
    int z_size = lstm->input_size + lstm->hidden_size;
    double z[MAX_COL_SIZE];
    
    for (int i=0; i<lstm->input_size; i++) {
        z[i] = input[i];
    }
    for (int i=0; i<lstm->hidden_size; i++) {
        z[lstm->input_size + i] = lstm->hidden_state[i];
    }

    // Calcul des portes
    double f[MAX_HIDDEN_SIZE];
    double i_gate[MAX_HIDDEN_SIZE];
    double g[MAX_HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    double o[MAX_HIDDEN_SIZE];

    double wfz[MAX_HIDDEN_SIZE];
    double wiz[MAX_HIDDEN_SIZE];
    double wcz[MAX_HIDDEN_SIZE];
    double woz[MAX_HIDDEN_SIZE];
    
    matrix_vector_product((double**)lstm->wf, z, lstm->hidden_size, z_size, wfz);
    matrix_vector_product((double**)lstm->wi, z, lstm->hidden_size, z_size, wiz);
    matrix_vector_product((double**)lstm->wc, z, lstm->hidden_size, z_size, wcz);
    matrix_vector_product((double**)lstm->wo, z, lstm->hidden_size, z_size, woz);

    for (int j=0; j<lstm->hidden_size; j++) {
        f[j] = sigmoid(wfz[j] + lstm->bf[j]);
        i_gate[j] = sigmoid(wiz[j] + lstm->bi[j]);
        g[j] = tanh(wcz[j] + lstm->bc[j]);
        o[j] = sigmoid(woz[j] + lstm->bo[j]);
    }

    // Mise à jour de la mémoire long-terme (cellule)
    for (int j=0; j<lstm->hidden_size; j++) {
        lstm->cell_state[j] = f[j] * lstm->cell_state[j] + i_gate[j] * g[j];
    }

    // Mise à jour de la mémoire court-terme (état caché)
    for (int j=0; j<lstm->hidden_size; j++) {
        lstm->hidden_state[j] = o[j] * tanh(lstm->cell_state[j]);
    }
}