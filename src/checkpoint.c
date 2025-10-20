#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "checkpoint.h"

static int write_exact(FILE* f, const void* buf, size_t sz) {
    return fwrite(buf, 1, sz, f) == sz ? 0 : -1;
}

static int read_exact(FILE* f, void* buf, size_t sz) {
    return fread(buf, 1, sz, f) == sz ? 0 : -1;
}

int saveLSTMCheckpoint(const char* path, const LSTM* net, uint64_t step, uint64_t rng_seed) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    LSTMCheckpointHeader h;
    memcpy(h.magic, "LSTMBIN\0", 8);
    h.version = 1u;
    h.inputSize = (uint32_t)net->inputSize;
    h.hiddenSize = (uint32_t)net->hiddenSize;
    h.step = step;
    h.rng_seed = rng_seed;

    if (write_exact(f, &h, sizeof(h)) != 0) { fclose(f); return -1; }

    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (write_exact(f, net->Wf[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (write_exact(f, net->Wi[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (write_exact(f, net->Wc[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (write_exact(f, net->Wo[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }

    if (write_exact(f, net->Bf, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (write_exact(f, net->Bi, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (write_exact(f, net->Bc, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (write_exact(f, net->Bo, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }

    if (write_exact(f, net->hiddenState, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (write_exact(f, net->cellState, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }

    fclose(f);
    return 0;
}

int loadLSTMCheckpoint(const char* path, LSTM* net, uint64_t* step, uint64_t* rng_seed) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    LSTMCheckpointHeader h;
    if (read_exact(f, &h, sizeof(h)) != 0) { fclose(f); return -1; }
    if (memcmp(h.magic, "LSTMBIN\0", 8) != 0) { fclose(f); return -1; }
    if (h.version != 1u) { fclose(f); return -1; }
    if (h.inputSize != (uint32_t)net->inputSize || h.hiddenSize != (uint32_t)net->hiddenSize) { fclose(f); return -1; }
    if (step) *step = h.step;
    if (rng_seed) *rng_seed = h.rng_seed;

    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wf[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wi[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wc[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wo[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    }

    if (read_exact(f, net->Bf, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (read_exact(f, net->Bi, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (read_exact(f, net->Bc, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (read_exact(f, net->Bo, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }

    if (read_exact(f, net->hiddenState, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }
    if (read_exact(f, net->cellState, sizeof(double)*net->hiddenSize) != 0) { fclose(f); return -1; }

    fclose(f);
    return 0;
}

LSTM* loadLSTM(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    LSTMCheckpointHeader h;
    if (read_exact(f, &h, sizeof(h)) != 0) { fclose(f); return NULL; }
    if (memcmp(h.magic, "LSTMBIN\0", 8) != 0) { fclose(f); return NULL; }
    if (h.version != 1u) { fclose(f); return NULL; }

    LSTM* net = initLSTM((int)h.inputSize, (int)h.hiddenSize);
    if (!net) { fclose(f); return NULL; }

    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wf[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wi[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wc[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    }
    for (int i=0;i<net->inputSize + net->hiddenSize;i++) {
        if (read_exact(f, net->Wo[i], sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    }

    if (read_exact(f, net->Bf, sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    if (read_exact(f, net->Bi, sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    if (read_exact(f, net->Bc, sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    if (read_exact(f, net->Bo, sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }

    if (read_exact(f, net->hiddenState, sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }
    if (read_exact(f, net->cellState, sizeof(double)*net->hiddenSize) != 0) { fclose(f); freeLSTM(net); return NULL; }

    fclose(f);
    return net;
}
