#ifndef LSTM_H
#define LSTM_H

#include "ppo.h"
#include "config.h"

typedef struct Lstm Lstm;

struct Lstm {
    int input_size;
    int hidden_size;

    double hidden_state[HIDDEN_SIZE];   // mémoire à court terme
    double cell_state[HIDDEN_SIZE];     // mémoire à long terme

    double wf[HIDDEN_SIZE][COL_SIZE];            // poids pour la forget gate    (quelle information oublier)
    double wi[HIDDEN_SIZE][COL_SIZE];            // poids pour la input gate     (quels candidats ajouter à la mémoire à long terme)
    double wc[HIDDEN_SIZE][COL_SIZE];            // poids pour la cell gate      (création de candidat pour la mémoire à long terme)
    double wo[HIDDEN_SIZE][COL_SIZE];            // poids pour la output gate    (que renvoyer en sortie)

    // Premier moment pour Adam (moyenne des gradients)
    double wf_m[HIDDEN_SIZE][COL_SIZE];
    double wi_m[HIDDEN_SIZE][COL_SIZE];
    double wc_m[HIDDEN_SIZE][COL_SIZE];
    double wo_m[HIDDEN_SIZE][COL_SIZE];

    // Second moment pour Adam (variance/moyenne des carrés des gradients)
    double wf_v[HIDDEN_SIZE][COL_SIZE];
    double wi_v[HIDDEN_SIZE][COL_SIZE];
    double wc_v[HIDDEN_SIZE][COL_SIZE];
    double wo_v[HIDDEN_SIZE][COL_SIZE];

    // Biais
    double bf[HIDDEN_SIZE];
    double bi[HIDDEN_SIZE];
    double bc[HIDDEN_SIZE];
    double bo[HIDDEN_SIZE];

    double bf_m[HIDDEN_SIZE];
    double bi_m[HIDDEN_SIZE];
    double bc_m[HIDDEN_SIZE];
    double bo_m[HIDDEN_SIZE];

    double bf_v[HIDDEN_SIZE];
    double bi_v[HIDDEN_SIZE];
    double bc_v[HIDDEN_SIZE];
    double bo_v[HIDDEN_SIZE];
};

void init_lstm(Lstm* lstm, int input_size, int hidden_size);
void lstm_forward(Lstm* lstm, Trajectory* traj, double input[INPUT_SIZE], int t);
void lstm_backward(Lstm* lstm, Trajectory* traj, double dL_dh_v[BATCH_SIZE][HIDDEN_SIZE], double dL_dh_p[BATCH_SIZE][HIDDEN_SIZE], double c_ini[HIDDEN_SIZE], double dL_dwf[HIDDEN_SIZE][COL_SIZE], double dL_dwi[HIDDEN_SIZE][COL_SIZE], double dL_dwc[HIDDEN_SIZE][COL_SIZE], double dL_dwo[HIDDEN_SIZE][COL_SIZE], double dL_dbf[HIDDEN_SIZE], double dL_dbi[HIDDEN_SIZE], double dL_dbc[HIDDEN_SIZE], double dL_dbo[HIDDEN_SIZE]);

#endif