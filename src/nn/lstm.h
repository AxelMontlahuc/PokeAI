#ifndef LSTM_H
#define LSTM_H

typedef struct Lstm Lstm;

struct Lstm {
    int input_size;
    int hidden_size;

    double* hidden_state;   // mémoire à court terme
    double* cell_state;     // mémoire à long terme

    double** wf;            // poids pour la forget gate    (quelle information oublier)
    double** wi;            // poids pour la input gate     (quels candidats ajouter à la mémoire à long terme)
    double** wc;            // poids pour la cell gate      (création de candidat pour la mémoire à long terme)
    double** wo;            // poids pour la output gate    (que renvoyer en sortie)

    // Premier moment pour Adam (moyenne des gradients)
    double** wf_m;
    double** wi_m;
    double** wc_m;
    double** wo_m;

    // Second moment pour Adam (variance/moyenne des carrés des gradients)
    double** wf_v;
    double** wi_v;
    double** wc_v;
    double** wo_v;

    // Biais
    double* bf;
    double* bi;
    double* bc;
    double* bo;

    double* bf_m;
    double* bi_m;
    double* bc_m;
    double* bo_m;

    double* bf_v;
    double* bi_v;
    double* bc_v;
    double* bo_v;
};

Lstm* init_lstm(int input_size, int hidden_size);
void free_lstm(Lstm* lstm);
void lstm_forward(Lstm* lstm, double* input);

#endif