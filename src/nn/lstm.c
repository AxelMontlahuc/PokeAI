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
float rand_normal() {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 < 1e-10f) u1 = 1e-10f; // Pas de log(0)
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// Produit scalaire canonique de R^n
float dot_product(float* x, float* y, int n) {
    float res = 0.0f;
    for (int i = 0; i < n; i++) {
        res += x[i] * y[i];
    }
    return res;
}

// Norme euclidienne associée au produit scalaire canonique
float norm(float* x, int n) {
    return sqrtf(dot_product(x, x, n));
}

// Extraction d'une colonne d'une matrice
void column(float matrix[HIDDEN_SIZE][HIDDEN_SIZE], int j, int rows, float* col) {
    for (int i = 0; i < rows; i++) {
        col[i] = matrix[i][j];
    }
}

// Initialisation orthogonale : on prend la décomposition QR d'une matrice aléatoire et on utilise Q
void orthogonal_init(float matrix[HIDDEN_SIZE][HIDDEN_SIZE], int rows, int cols) {
    float projection[HIDDEN_SIZE];
    float column_j_buf[HIDDEN_SIZE];
    float column_k_buf[HIDDEN_SIZE];
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand_normal();
        }
    }
    
    // Gram-Schmidt
    for (int j=0; j<cols; j++) {
        // Orthogonalisation
        for (int i=0; i<rows; i++) {
            projection[i] = 0.0f;
        }

        column(matrix, j, rows, column_j_buf);

        for (int k=0; k<j; k++) {
            column(matrix, k, rows, column_k_buf);

            float dot = dot_product(column_j_buf, column_k_buf, rows);
            for (int i=0; i<rows; i++) {
                projection[i] += dot * matrix[i][k];
            }
        }

        for (int i=0; i<rows; i++) {
            matrix[i][j] -= projection[i];
        }

        column(matrix, j, rows, column_j_buf);

        // Normalisation
        float n = norm(column_j_buf, rows);
        if (n > 1e-6f) { // Pas de division par zéro
            for (int i=0; i<rows; i++) {
                matrix[i][j] /= n;
            }
        }
    }
}

// Initialisation de Xavier : distribution uniforme dans [-limit, limit] avec limit = sqrt(6 / (input_size + hidden_size))
static void xavier_init_lstm(float matrix[HIDDEN_SIZE][INPUT_SIZE], int input_size, int hidden_size) {
    float limit = sqrtf(6.0f / (float)(input_size + hidden_size));
    for (int i=0; i<hidden_size; i++) {
        for (int j=0; j<input_size; j++) {
            float r = (float)rand() / (float)RAND_MAX;
            float val = (r * 2.0f - 1.0f) * limit;
            matrix[i][j] = val;
        }
    }
}

// Fonction auxiliaire pour l'initialisation des poids
void init_weights(float matrix[HIDDEN_SIZE][COL_SIZE], int input_size, int hidden_size) {
    float wx[HIDDEN_SIZE][INPUT_SIZE];
    float wh[HIDDEN_SIZE][HIDDEN_SIZE];

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
        lstm->hidden_state[i] = 0.0f;
        lstm->cell_state[i] = 0.0f;
    }

    for (int i = 0; i < hidden_size; i++) {
        lstm->bf[i] = 0.0f;
        lstm->bi[i] = 0.0f;
        lstm->bc[i] = 0.0f;
        lstm->bo[i] = 0.0f;

        lstm->bf_m[i] = 0.0f;
        lstm->bi_m[i] = 0.0f;
        lstm->bc_m[i] = 0.0f;
        lstm->bo_m[i] = 0.0f;

        lstm->bf_v[i] = 0.0f;
        lstm->bi_v[i] = 0.0f;
        lstm->bc_v[i] = 0.0f;
        lstm->bo_v[i] = 0.0f;
    }

    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size + hidden_size; j++) {
            lstm->wf[i][j] = 0.0f;
            lstm->wi[i][j] = 0.0f;
            lstm->wc[i][j] = 0.0f;
            lstm->wo[i][j] = 0.0f;

            lstm->wf_m[i][j] = 0.0f;
            lstm->wi_m[i][j] = 0.0f;
            lstm->wc_m[i][j] = 0.0f;
            lstm->wo_m[i][j] = 0.0f;

            lstm->wf_v[i][j] = 0.0f;
            lstm->wi_v[i][j] = 0.0f;
            lstm->wc_v[i][j] = 0.0f;
            lstm->wo_v[i][j] = 0.0f;
        }
    }

    init_weights(lstm->wf, input_size, hidden_size);
    init_weights(lstm->wi, input_size, hidden_size);
    init_weights(lstm->wc, input_size, hidden_size);
    init_weights(lstm->wo, input_size, hidden_size);
}

// Fonction sigmoïde
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void matrix_vector_product(float matrix[HIDDEN_SIZE][COL_SIZE], float* vector, int rows, int cols, float* result) {
    for (int i=0; i<rows; i++) {
        result[i] = 0.0f;
        for (int j=0; j<cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Propagation
void lstm_forward(Lstm* lstm, Trajectory* traj, int input[INPUT_SIZE], int t) {
    // Concaténation de l'entrée et de l'état caché dans un array z
    int z_size = lstm->input_size + lstm->hidden_size;
    float z[COL_SIZE];
    
    for (int i=0; i<lstm->input_size; i++) {
        z[i] = (float)input[i];
    }
    for (int i=0; i<lstm->hidden_size; i++) {
        z[lstm->input_size + i] = lstm->hidden_state[i];
    }

    memcpy(traj->z[t], z, sizeof(float) * (size_t)z_size); // Stockage de z pour la rétropropagation

    // Calcul des portes
    float f[HIDDEN_SIZE];
    float i_gate[HIDDEN_SIZE];
    float g[HIDDEN_SIZE]; // On note g la porte candidat pour éviter la confusion avec la mémoire à long-terme (cellule) c
    float o[HIDDEN_SIZE];

    float wfz[HIDDEN_SIZE];
    float wiz[HIDDEN_SIZE];
    float wcz[HIDDEN_SIZE];
    float woz[HIDDEN_SIZE];
    
    matrix_vector_product(lstm->wf, z, lstm->hidden_size, z_size, wfz);
    matrix_vector_product(lstm->wi, z, lstm->hidden_size, z_size, wiz);
    matrix_vector_product(lstm->wc, z, lstm->hidden_size, z_size, wcz);
    matrix_vector_product(lstm->wo, z, lstm->hidden_size, z_size, woz);

    for (int j=0; j<lstm->hidden_size; j++) {
        f[j] = sigmoid(wfz[j] + lstm->bf[j]);
        i_gate[j] = sigmoid(wiz[j] + lstm->bi[j]);
        g[j] = tanhf(wcz[j] + lstm->bc[j]);
        o[j] = sigmoid(woz[j] + lstm->bo[j]);
    }

    // Mise à jour de la mémoire long-terme (cellule)
    for (int j=0; j<lstm->hidden_size; j++) {
        lstm->cell_state[j] = f[j] * lstm->cell_state[j] + i_gate[j] * g[j];
    }

    // Mise à jour de la mémoire court-terme (état caché)
    for (int j=0; j<lstm->hidden_size; j++) {
        lstm->hidden_state[j] = o[j] * tanhf(lstm->cell_state[j]);
    }

    // Stockage des portes pour la rétropropagation
    memcpy(traj->f[t], f, sizeof(float) * (size_t)lstm->hidden_size); // On cast à size_t aka unsigned long pour éviter les warnings
    memcpy(traj->i[t], i_gate, sizeof(float) * (size_t)lstm->hidden_size);
    memcpy(traj->g[t], g, sizeof(float) * (size_t)lstm->hidden_size);
    memcpy(traj->o[t], o, sizeof(float) * (size_t)lstm->hidden_size);
    memcpy(traj->hidden_states[t], lstm->hidden_state, sizeof(float) * (size_t)HIDDEN_SIZE);
    memcpy(traj->c[t], lstm->cell_state, sizeof(float) * (size_t)HIDDEN_SIZE);
}

void lstm_backward(Lstm* lstm, Minibatch* minibatch, float dL_dh_v[MINIBATCH_SIZE][HIDDEN_SIZE], float dL_dh_p[MINIBATCH_SIZE][HIDDEN_SIZE], float c_ini[NUM_SEQS][HIDDEN_SIZE], float dL_dwf[HIDDEN_SIZE][COL_SIZE], float dL_dwi[HIDDEN_SIZE][COL_SIZE], float dL_dwc[HIDDEN_SIZE][COL_SIZE], float dL_dwo[HIDDEN_SIZE][COL_SIZE], float dL_dbf[HIDDEN_SIZE], float dL_dbi[HIDDEN_SIZE], float dL_dbc[HIDDEN_SIZE], float dL_dbo[HIDDEN_SIZE]) {
    float dL_dh[MINIBATCH_SIZE][HIDDEN_SIZE] = {0};
    float dL_do[MINIBATCH_SIZE][HIDDEN_SIZE];
    float dL_dc[MINIBATCH_SIZE][HIDDEN_SIZE];
    float dL_df[MINIBATCH_SIZE][HIDDEN_SIZE];
    float dL_di[MINIBATCH_SIZE][HIDDEN_SIZE];
    float dL_dg[MINIBATCH_SIZE][HIDDEN_SIZE];

    for (int t=MINIBATCH_SIZE-1; t>=0; t--) {
        //dL/dh = dL/dh_value_head + dL/dh_policy_head
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_dh[t][j] += dL_dh_v[t][j] + dL_dh_p[t][j];
        }

        // dL_do = dL/dh * tanh(c) (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_do[t][j] = dL_dh[t][j] * tanhf(minibatch->c[t][j]);
        }

        // dL_dc = dL/dh * o * (1 - tanh(c)^2) + dL_dc_next
        for (int j=0; j<HIDDEN_SIZE; j++) {
            float dL_dc_next = (t % SEQ_LEN == SEQ_LEN - 1) ? 0.0f : dL_dc[t+1][j] * minibatch->f[t+1][j];
            float tcell = tanhf(minibatch->c[t][j]);
            dL_dc[t][j] = dL_dh[t][j] * minibatch->o[t][j] * (1.0f - tcell * tcell) + dL_dc_next;
        }

        // dL_df = dc * c_prev (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            float c_prev = (t % SEQ_LEN == 0) ? c_ini[t / SEQ_LEN][j] : minibatch->c[t-1][j];
            dL_df[t][j] = dL_dc[t][j] * c_prev;
        }

        // dL_di = dc * g (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_di[t][j] = dL_dc[t][j] * minibatch->g[t][j];
        }

        // dL_dg = dc * i (où * est le produit de Hadamard)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_dg[t][j] = dL_dc[t][j] * minibatch->i[t][j];
        }

        // Dérivation à travers les fonctions d'activation (sigmoïde et tanh)
        // Par souci de simplicité on fera l'abus de notation dL_df pour désigner dL/d(f pré-activation), etc...
        // dL/d(f pré-activation) = dL/df * f * (1 - f) (car f = sigmoid(f pré-activation)) (idem pour i et o) où * est le produit de Hadamard
        // dL_d(g pré-activation) = dL/dg * (1 - g^2) (car g = tanh(g pré-activation)) où * est le produit de Hadamard
        for (int j=0; j<HIDDEN_SIZE; j++) {
            dL_df[t][j] = dL_df[t][j] * minibatch->f[t][j] * (1.0f - minibatch->f[t][j]);
            dL_di[t][j] = dL_di[t][j] * minibatch->i[t][j] * (1.0f - minibatch->i[t][j]);
            dL_dg[t][j] = dL_dg[t][j] * (1.0f - minibatch->g[t][j] * minibatch->g[t][j]);
            dL_do[t][j] = dL_do[t][j] * minibatch->o[t][j] * (1.0f - minibatch->o[t][j]);
        }

        // Gradients pour les poids et les biais : dL/dw = dL/dporte * z et dL/db = dL/dporte (où porte est f, i, g ou o)
        for (int j=0; j<HIDDEN_SIZE; j++) {
            for (int k=0; k<COL_SIZE; k++) {
                dL_dwf[j][k] += dL_df[t][j] * minibatch->z[t][k];
                dL_dwi[j][k] += dL_di[t][j] * minibatch->z[t][k];
                dL_dwc[j][k] += dL_dg[t][j] * minibatch->z[t][k];
                dL_dwo[j][k] += dL_do[t][j] * minibatch->z[t][k];
            }

            dL_dbf[j] += dL_df[t][j];
            dL_dbi[j] += dL_di[t][j];
            dL_dbc[j] += dL_dg[t][j];
            dL_dbo[j] += dL_do[t][j];
        }

        // dL/dh = dL/dh_porte + dL/dh_reccurent, on calcule ici dL/dh_recurrent = dz[INPUT_SIZE:] où dz = Σ dL/dporte * w (où porte est f, i, g ou o)
        float dz[COL_SIZE] = {0};
        for (int k=0; k<COL_SIZE; k++) {
            for (int j=0; j<HIDDEN_SIZE; j++) {
                dz[k] += lstm->wf[j][k] * dL_df[t][j] + lstm->wi[j][k] * dL_di[t][j] + lstm->wc[j][k] * dL_dg[t][j] + lstm->wo[j][k] * dL_do[t][j];
            }
        }

        if (t % SEQ_LEN > 0) {
            for (int j=0; j<HIDDEN_SIZE; j++) {
                dL_dh[t-1][j] += dz[INPUT_SIZE + j];
            }
        }
    }

    // Calcul de la moyenne sur le minibatch
    for (int j=0; j<HIDDEN_SIZE; j++) {
        for (int k=0; k<COL_SIZE; k++) {
            dL_dwf[j][k] /= (float)MINIBATCH_SIZE;
            dL_dwi[j][k] /= (float)MINIBATCH_SIZE;
            dL_dwc[j][k] /= (float)MINIBATCH_SIZE;
            dL_dwo[j][k] /= (float)MINIBATCH_SIZE;
        }
        dL_dbf[j] /= (float)MINIBATCH_SIZE;
        dL_dbi[j] /= (float)MINIBATCH_SIZE;
        dL_dbc[j] /= (float)MINIBATCH_SIZE;
        dL_dbo[j] /= (float)MINIBATCH_SIZE;
    }
}

// Fonction auxiliaire pour recalculer les états cachés (hidden_states), ce qui est requis lors de la rétropropagation
void lstm_recompute_minibatch(Lstm* lstm, Minibatch* minibatch) {
    int z_size = lstm->input_size + lstm->hidden_size;

    for (int s = 0; s < NUM_SEQS; s++) {
        float current_h[HIDDEN_SIZE];
        float current_c[HIDDEN_SIZE];
        memcpy(current_h, minibatch->h_ini[s], sizeof(current_h));
        memcpy(current_c, minibatch->c_ini[s], sizeof(current_c));

        for (int step = 0; step < SEQ_LEN; step++) {
            int t = s * SEQ_LEN + step;
            float z[COL_SIZE];
            
            int normalized_state[INPUT_SIZE];
            float sumsq = 0.0f;
            for (int i = 0; i < 24; i++) {
                float v = (float)minibatch->states[t][i];
                sumsq += v * v;
            }
            float norm_val = sqrtf(sumsq) + 1e-8f;
            for (int i = 0; i < 24; i++) {
                float v = (float)minibatch->states[t][i] / norm_val;
                float scaled = v * 1000.0f;
                normalized_state[i] = (int)scaled;
            }
            for (int i = 24; i < INPUT_SIZE; i++) {
                normalized_state[i] = minibatch->states[t][i];
            }

            for (int i = 0; i < lstm->input_size; i++) {
                z[i] = (float)normalized_state[i];
            }
            for (int i = 0; i < lstm->hidden_size; i++) {
                z[lstm->input_size + i] = current_h[i];
            }
            memcpy(minibatch->z[t], z, sizeof(float) * (size_t)z_size);

            float f[HIDDEN_SIZE];
            float i_gate[HIDDEN_SIZE];
            float g[HIDDEN_SIZE];
            float o[HIDDEN_SIZE];

            float wfz[HIDDEN_SIZE];
            float wiz[HIDDEN_SIZE];
            float wcz[HIDDEN_SIZE];
            float woz[HIDDEN_SIZE];

            matrix_vector_product(lstm->wf, z, lstm->hidden_size, z_size, wfz);
            matrix_vector_product(lstm->wi, z, lstm->hidden_size, z_size, wiz);
            matrix_vector_product(lstm->wc, z, lstm->hidden_size, z_size, wcz);
            matrix_vector_product(lstm->wo, z, lstm->hidden_size, z_size, woz);

            for (int j = 0; j < lstm->hidden_size; j++) {
                f[j] = sigmoid(wfz[j] + lstm->bf[j]);
                i_gate[j] = sigmoid(wiz[j] + lstm->bi[j]);
                g[j] = tanhf(wcz[j] + lstm->bc[j]);
                o[j] = sigmoid(woz[j] + lstm->bo[j]);
            }

            for (int j = 0; j < lstm->hidden_size; j++) {
                current_c[j] = f[j] * current_c[j] + i_gate[j] * g[j];
            }

            for (int j = 0; j < lstm->hidden_size; j++) {
                current_h[j] = o[j] * tanhf(current_c[j]);
            }

            memcpy(minibatch->f[t], f, sizeof(float) * (size_t)lstm->hidden_size);
            memcpy(minibatch->i[t], i_gate, sizeof(float) * (size_t)lstm->hidden_size);
            memcpy(minibatch->g[t], g, sizeof(float) * (size_t)lstm->hidden_size);
            memcpy(minibatch->o[t], o, sizeof(float) * (size_t)lstm->hidden_size);
            memcpy(minibatch->hidden_states[t], current_h, sizeof(float) * (size_t)HIDDEN_SIZE);
            memcpy(minibatch->c[t], current_c, sizeof(float) * (size_t)HIDDEN_SIZE);
        }
    }
}