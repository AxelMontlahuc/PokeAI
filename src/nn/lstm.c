#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "lstm.h"
#include "ppo.h"
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
void column(double matrix[HIDDEN_SIZE][HIDDEN_SIZE], int j, int rows, double* col) {
    for (int i = 0; i < rows; i++) {
        col[i] = matrix[i][j];
    }
}

// Initialisation orthogonale : on prend la décomposition QR d'une matrice aléatoire et on utilise Q
void orthogonal_init(double matrix[HIDDEN_SIZE][HIDDEN_SIZE], int rows, int cols) {
    double projection[HIDDEN_SIZE];
    double column_j_buf[HIDDEN_SIZE];
    double column_k_buf[HIDDEN_SIZE];
    
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
static void xavier_init_lstm(double matrix[HIDDEN_SIZE][INPUT_SIZE], int input_size, int hidden_size) {
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
void init_weights(double matrix[HIDDEN_SIZE][COL_SIZE], int input_size, int hidden_size) {
    double wx[HIDDEN_SIZE][INPUT_SIZE];
    double wh[HIDDEN_SIZE][HIDDEN_SIZE];

    for (int i=0; i<hidden_size; i++) {
        for (int j=0; j<input_size; j++) {
            wx[i][j] = matrix[i][j];
        }
        for (int j=0; j<hidden_size; j++) {
            wh[i][j] = matrix[i][input_size + j];
        }
    }

    xavier_init_lstm(wx, input_size, hidden_size);
    orthogonal_init(wh, hidden_size, hidden_size);

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
    
    for (int i = 0; i < hidden_size; i++) {
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

void matrix_vector_product(double matrix[HIDDEN_SIZE][COL_SIZE], double* vector, int rows, int cols, double* result) {
    for (int i=0; i<rows; i++) {
        result[i] = 0.0;
        for (int j=0; j<cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Propagation
void lstm_forward(Lstm* lstm, Trajectory* traj, int input[INPUT_SIZE], int t) {
    // Concaténation de l'entrée et de l'état caché dans un array z
    int z_size = lstm->input_size + lstm->hidden_size;
    double z[COL_SIZE];
    
    for (int i=0; i<lstm->input_size; i++) {
        z[i] = (double)input[i];
    }
    for (int i=0; i<lstm->hidden_size; i++) {
        z[lstm->input_size + i] = lstm->hidden_state[i];
    }

    memcpy(traj->z[t], z, sizeof(double) * (size_t)z_size); // Stockage de z pour la rétropropagation

    // Calcul des portes
    double f[HIDDEN_SIZE];
    double i_gate[HIDDEN_SIZE];
    double g[HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    double o[HIDDEN_SIZE];

    double wfz[HIDDEN_SIZE];
    double wiz[HIDDEN_SIZE];
    double wcz[HIDDEN_SIZE];
    double woz[HIDDEN_SIZE];
    
    matrix_vector_product(lstm->wf, z, lstm->hidden_size, z_size, wfz);
    matrix_vector_product(lstm->wi, z, lstm->hidden_size, z_size, wiz);
    matrix_vector_product(lstm->wc, z, lstm->hidden_size, z_size, wcz);
    matrix_vector_product(lstm->wo, z, lstm->hidden_size, z_size, woz);

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

    // Stockage des portes pour la rétropropagation
    memcpy(traj->f[t], f, sizeof(double) * (size_t)lstm->hidden_size); // On cast à size_t aka unsigned long pour éviter les warnings
    memcpy(traj->i[t], i_gate, sizeof(double) * (size_t)lstm->hidden_size);
    memcpy(traj->g[t], g, sizeof(double) * (size_t)lstm->hidden_size);
    memcpy(traj->o[t], o, sizeof(double) * (size_t)lstm->hidden_size);
    memcpy(traj->hidden_states[t], lstm->hidden_state, sizeof(double) * (size_t)HIDDEN_SIZE);
    memcpy(traj->c[t], lstm->cell_state, sizeof(double) * (size_t)HIDDEN_SIZE);
}

// Rétropropagation
void lstm_backward(Lstm* lstm, Trajectory* traj, double dL_dh_v[BATCH_SIZE][HIDDEN_SIZE], double dL_dh_p[BATCH_SIZE][HIDDEN_SIZE], double c_ini[HIDDEN_SIZE], double dL_dwf[HIDDEN_SIZE][COL_SIZE], double dL_dwi[HIDDEN_SIZE][COL_SIZE], double dL_dwc[HIDDEN_SIZE][COL_SIZE], double dL_dwo[HIDDEN_SIZE][COL_SIZE], double dL_dbf[HIDDEN_SIZE], double dL_dbi[HIDDEN_SIZE], double dL_dbc[HIDDEN_SIZE], double dL_dbo[HIDDEN_SIZE]) {
    double dL_dh[BATCH_SIZE][HIDDEN_SIZE] = {0};
    double dL_do[BATCH_SIZE][HIDDEN_SIZE];
    double dL_dc[BATCH_SIZE][HIDDEN_SIZE];
    double dL_df[BATCH_SIZE][HIDDEN_SIZE];
    double dL_di[BATCH_SIZE][HIDDEN_SIZE];
    double dL_dg[BATCH_SIZE][HIDDEN_SIZE];

    for (int t=BATCH_SIZE-1; t>=0; t--) {
        //dL/dh = dL/dh_value_head + dL/dh_policy_head
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_dh[t][j] += dL_dh_v[t][j] + dL_dh_p[t][j];
        }

        // dL_do = dL/dh * tanh(c) (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_do[t][j] = dL_dh[t][j] * tanh(traj->c[t][j]);
        }

        // dL_dc = dL/dh * o * (1 - tanh(c)^2) + dL_dc_next
        for (int j=0; j<HIDDEN_SIZE; j++) {
            double dL_dc_next = (t == BATCH_SIZE-1) ? 0.0 : dL_dc[t+1][j] * traj->f[t+1][j];
            dL_dc[t][j] = dL_dh[t][j] * traj->o[t][j] * (1 - tanh(traj->c[t][j]) * tanh(traj->c[t][j])) + dL_dc_next;
        }

        // dL_df = dc * c_prev (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            double c_prev = (t == 0) ? c_ini[j] : traj->c[t-1][j];
            dL_df[t][j] = dL_dc[t][j] * c_prev;
        }

        // dL_di = dc * g (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_di[t][j] = dL_dc[t][j] * traj->g[t][j];
        }

        // dL_dg = dc * i (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_dg[t][j] = dL_dc[t][j] * traj->i[t][j];
        }

        // Dérivation à travers les fonctions d'activation (sigmoïde et tanh)
        // Par souci de simplicité on fera l'abus de notation dL_df pour désigner dL/d(f pré-activation), etc...
        // dL/d(f pré-activation) = dL/df * f * (1 - f) (car f = sigmoid(f pré-activation)) (idem pour i et o) où * est le produit de Hadamard
        // dL_d(g pré-activation) = dL/dg * (1 - g^2) (car g = tanh(g pré-activation)) où * est le produit de Hadamard
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_df[t][j] = dL_df[t][j] * traj->f[t][j] * (1 - traj->f[t][j]);
            dL_di[t][j] = dL_di[t][j] * traj->i[t][j] * (1 - traj->i[t][j]);
            dL_dg[t][j] = dL_dg[t][j] * (1 - traj->g[t][j] * traj->g[t][j]);
            dL_do[t][j] = dL_do[t][j] * traj->o[t][j] * (1 - traj->o[t][j]);
        }

        // Gradients pour les poids et les biais : dL/dw = dL/dporte * z et dL/db = dL/dporte (où porte est f, i, g ou o)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            for (int k=0; k<COL_SIZE; k++) {
                dL_dwf[j][k] += dL_df[t][j] * traj->z[t][k];
                dL_dwi[j][k] += dL_di[t][j] * traj->z[t][k];
                dL_dwc[j][k] += dL_dg[t][j] * traj->z[t][k];
                dL_dwo[j][k] += dL_do[t][j] * traj->z[t][k];
            }

            dL_dbf[j] += dL_df[t][j];
            dL_dbi[j] += dL_di[t][j];
            dL_dbc[j] += dL_dg[t][j];
            dL_dbo[j] += dL_do[t][j];
        }

        // dL/dh = dL/dh_porte + dL/dh_reccurent, on calcule ici dL/dh_recurrent = dz[INPUT_SIZE:] où dz = Σ dL/dporte * w (où porte est f, i, g ou o)
        double dz[COL_SIZE] = {0};
        for (int k=0; k<COL_SIZE; k++) {
            for (int j=0; j<HIDDEN_SIZE; j++) {
                dz[k] += lstm->wf[j][k] * dL_df[t][j] + lstm->wi[j][k] * dL_di[t][j] + lstm->wc[j][k] * dL_dg[t][j] + lstm->wo[j][k] * dL_do[t][j];
            }
        }

        if (t > 0) {
            for (int j=0; j<HIDDEN_SIZE; j++) {
                dL_dh[t-1][j] += dz[INPUT_SIZE + j];
            }
        }
    }
}