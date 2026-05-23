#ifndef LSTM_H
#define LSTM_H

#include "ppo.h"
#include "config.h"

typedef struct Lstm Lstm;

struct Lstm {
    int input_size;
    int hidden_size;

    float hidden_state[HIDDEN_SIZE];   // mémoire à court terme
    float cell_state[HIDDEN_SIZE];     // mémoire à long terme

    float wf[HIDDEN_SIZE][COL_SIZE];            // poids pour la forget gate    (quelle information oublier)
    float wi[HIDDEN_SIZE][COL_SIZE];            // poids pour la input gate     (quels candidats ajouter à la mémoire à long terme)
    float wc[HIDDEN_SIZE][COL_SIZE];            // poids pour la cell gate      (création de candidat pour la mémoire à long terme)
    float wo[HIDDEN_SIZE][COL_SIZE];            // poids pour la output gate    (que renvoyer en sortie)

    // Premier moment pour Adam (moyenne des gradients)
    float wf_m[HIDDEN_SIZE][COL_SIZE];
    float wi_m[HIDDEN_SIZE][COL_SIZE];
    float wc_m[HIDDEN_SIZE][COL_SIZE];
    float wo_m[HIDDEN_SIZE][COL_SIZE];

    // Second moment pour Adam (variance/moyenne des carrés des gradients)
    float wf_v[HIDDEN_SIZE][COL_SIZE];
    float wi_v[HIDDEN_SIZE][COL_SIZE];
    float wc_v[HIDDEN_SIZE][COL_SIZE];
    float wo_v[HIDDEN_SIZE][COL_SIZE];

    // Biais
    float bf[HIDDEN_SIZE];
    float bi[HIDDEN_SIZE];
    float bc[HIDDEN_SIZE];
    float bo[HIDDEN_SIZE];

    float bf_m[HIDDEN_SIZE];
    float bi_m[HIDDEN_SIZE];
    float bc_m[HIDDEN_SIZE];
    float bo_m[HIDDEN_SIZE];

    float bf_v[HIDDEN_SIZE];
    float bi_v[HIDDEN_SIZE];
    float bc_v[HIDDEN_SIZE];
    float bo_v[HIDDEN_SIZE];
};

void init_lstm(Lstm* lstm, int input_size, int hidden_size);
void lstm_forward(Lstm* lstm, Trajectory* traj, int input[INPUT_SIZE], int t);
void lstm_backward(Lstm* lstm, Minibatch* minibatch, float dL_dh_v[MINIBATCH_SIZE][HIDDEN_SIZE], float dL_dh_p[MINIBATCH_SIZE][HIDDEN_SIZE], float c_ini[NUM_SEQS][HIDDEN_SIZE], float dL_dwf[HIDDEN_SIZE][COL_SIZE], float dL_dwi[HIDDEN_SIZE][COL_SIZE], float dL_dwc[HIDDEN_SIZE][COL_SIZE], float dL_dwo[HIDDEN_SIZE][COL_SIZE], float dL_dbf[HIDDEN_SIZE], float dL_dbi[HIDDEN_SIZE], float dL_dbc[HIDDEN_SIZE], float dL_dbo[HIDDEN_SIZE]);
void lstm_recompute_minibatch(Lstm* lstm, Minibatch* minibatch);

#endif