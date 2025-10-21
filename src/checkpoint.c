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

    char magic[8];
    memcpy(magic, "LSTMBIN\0", 8);
    uint32_t version = 2u;
    uint32_t inputSize = (uint32_t)net->inputSize;
    uint32_t hiddenSize = (uint32_t)net->hiddenSize;
    uint64_t episodes = step;
    uint64_t seed = rng_seed;

    if (write_exact(f, magic, sizeof(magic)) != 0) { fclose(f); return -1; }
    if (write_exact(f, &version, sizeof(version)) != 0) { fclose(f); return -1; }
    if (write_exact(f, &inputSize, sizeof(inputSize)) != 0) { fclose(f); return -1; }
    if (write_exact(f, &hiddenSize, sizeof(hiddenSize)) != 0) { fclose(f); return -1; }
    if (write_exact(f, &episodes, sizeof(episodes)) != 0) { fclose(f); return -1; }
    if (write_exact(f, &seed, sizeof(seed)) != 0) { fclose(f); return -1; }

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

    char magic[8];
    uint32_t version;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint64_t episodes_or_step;
    uint64_t seed;

    if (read_exact(f, magic, sizeof(magic)) != 0) { fclose(f); return -1; }
    if (memcmp(magic, "LSTMBIN\0", 8) != 0) { fclose(f); return -1; }
    if (read_exact(f, &version, sizeof(version)) != 0) { fclose(f); return -1; }
    if (version != 1u && version != 2u) { fclose(f); return -1; }
    if (read_exact(f, &inputSize, sizeof(inputSize)) != 0) { fclose(f); return -1; }
    if (read_exact(f, &hiddenSize, sizeof(hiddenSize)) != 0) { fclose(f); return -1; }
    if (read_exact(f, &episodes_or_step, sizeof(episodes_or_step)) != 0) { fclose(f); return -1; }
    if (read_exact(f, &seed, sizeof(seed)) != 0) { fclose(f); return -1; }

    if (inputSize != (uint32_t)net->inputSize || hiddenSize != (uint32_t)net->hiddenSize) { fclose(f); return -1; }
    if (step) *step = episodes_or_step;
    if (rng_seed) *rng_seed = seed;

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

LSTM* loadLSTM(const char* path, uint64_t* episodes, uint64_t* rng_seed) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[8];
    uint32_t version;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint64_t episodes_or_step;
    uint64_t seed;

    if (read_exact(f, magic, sizeof(magic)) != 0) { fclose(f); return NULL; }
    if (memcmp(magic, "LSTMBIN\0", 8) != 0) { fclose(f); return NULL; }
    if (read_exact(f, &version, sizeof(version)) != 0) { fclose(f); return NULL; }
    if (version != 1u && version != 2u) { fclose(f); return NULL; }
    if (read_exact(f, &inputSize, sizeof(inputSize)) != 0) { fclose(f); return NULL; }
    if (read_exact(f, &hiddenSize, sizeof(hiddenSize)) != 0) { fclose(f); return NULL; }
    if (read_exact(f, &episodes_or_step, sizeof(episodes_or_step)) != 0) { fclose(f); return NULL; }
    if (read_exact(f, &seed, sizeof(seed)) != 0) { fclose(f); return NULL; }

    LSTM* net = initLSTM((int)inputSize, (int)hiddenSize);
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
    if (episodes) *episodes = episodes_or_step;
    if (rng_seed) *rng_seed = seed;
    return net;
}
