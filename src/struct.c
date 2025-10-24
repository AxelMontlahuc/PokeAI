#include "struct.h"

double heInitialization(double fanIn) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(2.0 / fanIn);
}

double xavierInitialization(double fanIn, double fanOut) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(6.0 / (fanIn + fanOut));
}

LSTM* initLSTM(int inputSize, int hiddenSize, int outputSize) {
    LSTM* network = malloc(sizeof(LSTM));
    assert(network != NULL);

    network->inputSize = inputSize;
    network->hiddenSize = hiddenSize;
    network->outputSize = outputSize;

    network->hiddenState = calloc(hiddenSize, sizeof(double));
    network->cellState = calloc(hiddenSize, sizeof(double));
    network->logits = calloc(network->outputSize, sizeof(double));
    network->probs = calloc(network->outputSize, sizeof(double));
    network->last_value = 0.0;
    assert(network->hiddenState != NULL && network->cellState != NULL && network->logits != NULL && network->probs != NULL);

    network->Wf = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wi = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wc = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wo = malloc((inputSize + hiddenSize) * sizeof(double*));
    assert(network->Wf != NULL && network->Wi != NULL && network->Wc != NULL && network->Wo != NULL);
    
    for (int i=0; i<(inputSize + hiddenSize); i++) {
        network->Wf[i] = malloc(hiddenSize * sizeof(double));
        network->Wi[i] = malloc(hiddenSize * sizeof(double));
        network->Wc[i] = malloc(hiddenSize * sizeof(double));
        network->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(network->Wf[i] != NULL && network->Wi[i] != NULL && network->Wc[i] != NULL && network->Wo[i] != NULL);

        for (int j=0; j<hiddenSize; j++) {
            network->Wf[i][j] = xavierInitialization(inputSize, hiddenSize);
            network->Wi[i][j] = xavierInitialization(inputSize, hiddenSize);
            network->Wc[i][j] = xavierInitialization(inputSize, hiddenSize);
            network->Wo[i][j] = xavierInitialization(inputSize, hiddenSize);
        }
    }

    network->Bf = malloc(hiddenSize * sizeof(double));
    network->Bi = malloc(hiddenSize * sizeof(double));
    network->Bc = malloc(hiddenSize * sizeof(double));
    network->Bo = malloc(hiddenSize * sizeof(double));
    assert(network->Bf != NULL && network->Bi != NULL && network->Bc != NULL && network->Bo != NULL);

    for (int i=0; i<hiddenSize; i++) {
        network->Bf[i] = 1.0;
        network->Bi[i] = 0.0;
        network->Bc[i] = 0.0;
        network->Bo[i] = 0.0;
    }

    network->Wout = malloc(network->hiddenSize * sizeof(double*));
    assert(network->Wout != NULL);
    for (int i=0; i<network->hiddenSize; i++) {
    network->Wout[i] = malloc(network->outputSize * sizeof(double));
        assert(network->Wout[i] != NULL);
        for (int j=0; j<network->outputSize; j++) {
            network->Wout[i][j] = xavierInitialization(network->hiddenSize, network->outputSize);
        }
    }
    network->Bout = calloc(network->outputSize, sizeof(double));
    assert(network->Bout != NULL);

    // Value head parameters
    network->Wv = malloc(hiddenSize * sizeof(double));
    assert(network->Wv != NULL);
    for (int j = 0; j < hiddenSize; j++) network->Wv[j] = xavierInitialization(hiddenSize, 1);
    network->Bv = 0.0;

    network->adam_t = 0;

    int Z = inputSize + hiddenSize;

    network->Wf_m = malloc(Z * sizeof(double*));
    network->Wf_v = malloc(Z * sizeof(double*));
    network->Wi_m = malloc(Z * sizeof(double*));
    network->Wi_v = malloc(Z * sizeof(double*));
    network->Wc_m = malloc(Z * sizeof(double*));
    network->Wc_v = malloc(Z * sizeof(double*));
    network->Wo_m = malloc(Z * sizeof(double*));
    network->Wo_v = malloc(Z * sizeof(double*));
    assert(network->Wf_m && network->Wf_v && network->Wi_m && network->Wi_v && network->Wc_m && network->Wc_v && network->Wo_m && network->Wo_v);
    
    for (int a=0; a<Z; a++) {
        network->Wf_m[a] = calloc(hiddenSize, sizeof(double));
        network->Wf_v[a] = calloc(hiddenSize, sizeof(double));
        network->Wi_m[a] = calloc(hiddenSize, sizeof(double));
        network->Wi_v[a] = calloc(hiddenSize, sizeof(double));
        network->Wc_m[a] = calloc(hiddenSize, sizeof(double));
        network->Wc_v[a] = calloc(hiddenSize, sizeof(double));
        network->Wo_m[a] = calloc(hiddenSize, sizeof(double));
        network->Wo_v[a] = calloc(hiddenSize, sizeof(double));
        assert(network->Wf_m[a] && network->Wf_v[a] && network->Wi_m[a] && network->Wi_v[a] && network->Wc_m[a] && network->Wc_v[a] && network->Wo_m[a] && network->Wo_v[a]);
    }

    network->Bf_m = calloc(hiddenSize, sizeof(double));
    network->Bf_v = calloc(hiddenSize, sizeof(double));
    network->Bi_m = calloc(hiddenSize, sizeof(double));
    network->Bi_v = calloc(hiddenSize, sizeof(double));
    network->Bc_m = calloc(hiddenSize, sizeof(double));
    network->Bc_v = calloc(hiddenSize, sizeof(double));
    network->Bo_m = calloc(hiddenSize, sizeof(double));
    network->Bo_v = calloc(hiddenSize, sizeof(double));
    assert(network->Bf_m && network->Bf_v && network->Bi_m && network->Bi_v && network->Bc_m && network->Bc_v && network->Bo_m && network->Bo_v);

    network->Wout_m = malloc(network->hiddenSize * sizeof(double*));
    network->Wout_v = malloc(network->hiddenSize * sizeof(double*));
    assert(network->Wout_m && network->Wout_v);
    for (int i=0; i<network->hiddenSize; i++) {
        network->Wout_m[i] = calloc(network->outputSize, sizeof(double));
        network->Wout_v[i] = calloc(network->outputSize, sizeof(double));
        assert(network->Wout_m[i] && network->Wout_v[i]);
    }
    network->Bout_m = calloc(network->outputSize, sizeof(double));
    network->Bout_v = calloc(network->outputSize, sizeof(double));
    assert(network->Bout_m && network->Bout_v);

    // Adam states for value head
    network->Wv_m = calloc(hiddenSize, sizeof(double));
    network->Wv_v = calloc(hiddenSize, sizeof(double));
    assert(network->Wv_m && network->Wv_v);
    network->Bv_m = 0.0;
    network->Bv_v = 0.0;

    return network;
}

void freeLSTM(LSTM* network) {
    free(network->hiddenState);
    free(network->cellState);
    free(network->logits);
    free(network->probs);

    for (int i=0; i<(network->inputSize + network->hiddenSize); i++) {
        free(network->Wf[i]);
        free(network->Wi[i]);
        free(network->Wc[i]);
        free(network->Wo[i]);
    }

    free(network->Wf);
    free(network->Wi);
    free(network->Wc);
    free(network->Wo);

    free(network->Bf);
    free(network->Bi);
    free(network->Bc);
    free(network->Bo);

    for (int i=0; i<network->hiddenSize; i++) {
        free(network->Wout[i]);
    }
    free(network->Wout);
    free(network->Bout);
    free(network->Wv);

    int Z = network->inputSize + network->hiddenSize;
    for (int a=0; a<Z; a++) {
        free(network->Wf_m[a]);
        free(network->Wf_v[a]);
        free(network->Wi_m[a]);
        free(network->Wi_v[a]);
        free(network->Wc_m[a]);
        free(network->Wc_v[a]);
        free(network->Wo_m[a]);
        free(network->Wo_v[a]);
    }
    free(network->Wf_m);
    free(network->Wf_v);
    free(network->Wi_m);
    free(network->Wi_v);
    free(network->Wc_m);
    free(network->Wc_v);
    free(network->Wo_m);
    free(network->Wo_v);

    free(network->Bf_m);
    free(network->Bf_v);
    free(network->Bi_m);
    free(network->Bi_v);
    free(network->Bc_m);
    free(network->Bc_v);
    free(network->Bo_m);
    free(network->Bo_v);

    for (int i=0; i<network->hiddenSize; i++) {
        free(network->Wout_m[i]);
        free(network->Wout_v[i]);
    }
    free(network->Wout_m);
    free(network->Wout_v);
    free(network->Bout_m);
    free(network->Bout_v);
    free(network->Wv_m);
    free(network->Wv_v);

    free(network);
}

trajectory* initTrajectory(int steps) {
    trajectory* traj = malloc(sizeof(trajectory));
    assert(traj != NULL);

    traj->states = malloc(steps * sizeof(state));
    traj->actions = malloc(steps * sizeof(MGBAButton));
    traj->rewards = malloc(steps * sizeof(double));
    traj->probs = malloc(steps * sizeof(double*));
    traj->behav_probs = malloc(steps * sizeof(double*));
    traj->values = malloc(steps * sizeof(double));
    assert(traj->states != NULL && traj->actions != NULL && traj->rewards != NULL && traj->probs != NULL && traj->behav_probs != NULL && traj->values != NULL);

    for (int i=0; i<steps; i++) {
        traj->probs[i] = NULL;
        traj->behav_probs[i] = NULL;
    }

    traj->steps = steps;

    return traj;
}

void freeTrajectory(trajectory* traj) {
    free(traj->states);
    for (int i=0; i<traj->steps; i++) {
        free(traj->probs[i]);
        free(traj->behav_probs[i]);
    }
    free(traj->actions);
    free(traj->rewards);
    free(traj->probs);
    free(traj->behav_probs);
    free(traj->values);
    free(traj);
}