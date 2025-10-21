#include "struct.h"

void freeState(state s) {
    mgba_free_map(s.map0);
    mgba_free_map(s.map1);
    mgba_free_map(s.map2);
    mgba_free_map(s.map3);

    free(s.team);
    free(s.enemy);
    free(s.PP);
}

double heInitialization(double fanIn) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(2.0 / fanIn);
}

double xavierInitialization(double fanIn, double fanOut) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(6.0 / (fanIn + fanOut));
}

LSTM* initLSTM(int inputSize, int hiddenSize) {
    LSTM* network = malloc(sizeof(LSTM));
    assert(network != NULL);

    network->inputSize = inputSize;
    network->hiddenSize = hiddenSize;

    network->hiddenState = calloc(hiddenSize, sizeof(double));
    network->cellState = calloc(hiddenSize, sizeof(double));
    assert(network->hiddenState != NULL && network->cellState != NULL);

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

    return network;
}

void freeLSTM(LSTM* network) {
    free(network->hiddenState);
    free(network->cellState);

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

    free(network);
}

trajectory* initTrajectory(int steps) {
    trajectory* traj = malloc(sizeof(trajectory));
    assert(traj != NULL);

    traj->states = malloc(steps * sizeof(state));
    traj->actions = malloc(steps * sizeof(MGBAButton));
    traj->rewards = malloc(steps * sizeof(double));
    traj->probs = malloc(steps * sizeof(double*));
    assert(traj->states != NULL && traj->actions != NULL && traj->rewards != NULL && traj->probs != NULL);

    for (int i=0; i<steps; i++) {
        traj->probs[i] = NULL;
    }

    traj->steps = steps;

    return traj;
}

void freeTrajectory(trajectory* traj) {
    for (int i=0; i<traj->steps; i++) {
        freeState(traj->states[i]);
    }
    free(traj->states);
    for (int i=0; i<traj->steps; i++) {
        free(traj->probs[i]);
    }
    free(traj->actions);
    free(traj->rewards);
    free(traj->probs);
    free(traj);
}