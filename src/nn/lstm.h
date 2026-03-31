#ifndef LSTM_H
#define LSTM_H

#include "config.h"

typedef struct Lstm Lstm;

struct Lstm {
    int input_size;
    int hidden_size;

    double hidden_state[MAX_HIDDEN_SIZE];   // mémoire à court terme
    double cell_state[MAX_HIDDEN_SIZE];     // mémoire à long terme

    double wf[MAX_HIDDEN_SIZE][MAX_COL_SIZE];            // poids pour la forget gate    (quelle information oublier)
    double wi[MAX_HIDDEN_SIZE][MAX_COL_SIZE];            // poids pour la input gate     (quels candidats ajouter à la mémoire à long terme)
    double wc[MAX_HIDDEN_SIZE][MAX_COL_SIZE];            // poids pour la cell gate      (création de candidat pour la mémoire à long terme)
    double wo[MAX_HIDDEN_SIZE][MAX_COL_SIZE];            // poids pour la output gate    (que renvoyer en sortie)

    // Premier moment pour Adam (moyenne des gradients)
    double wf_m[MAX_HIDDEN_SIZE][MAX_COL_SIZE];
    double wi_m[MAX_HIDDEN_SIZE][MAX_COL_SIZE];
    double wc_m[MAX_HIDDEN_SIZE][MAX_COL_SIZE];
    double wo_m[MAX_HIDDEN_SIZE][MAX_COL_SIZE];

    // Second moment pour Adam (variance/moyenne des carrés des gradients)
    double wf_v[MAX_HIDDEN_SIZE][MAX_COL_SIZE];
    double wi_v[MAX_HIDDEN_SIZE][MAX_COL_SIZE];
    double wc_v[MAX_HIDDEN_SIZE][MAX_COL_SIZE];
    double wo_v[MAX_HIDDEN_SIZE][MAX_COL_SIZE];

    // Biais
    double bf[MAX_HIDDEN_SIZE];
    double bi[MAX_HIDDEN_SIZE];
    double bc[MAX_HIDDEN_SIZE];
    double bo[MAX_HIDDEN_SIZE];

    double bf_m[MAX_HIDDEN_SIZE];
    double bi_m[MAX_HIDDEN_SIZE];
    double bc_m[MAX_HIDDEN_SIZE];
    double bo_m[MAX_HIDDEN_SIZE];

    double bf_v[MAX_HIDDEN_SIZE];
    double bi_v[MAX_HIDDEN_SIZE];
    double bc_v[MAX_HIDDEN_SIZE];
    double bo_v[MAX_HIDDEN_SIZE];
};

void init_lstm(Lstm* lstm, int input_size, int hidden_size);
void lstm_forward(Lstm* lstm, double* input);

#endif