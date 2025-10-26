#include "checkpoint.h"

int saveLSTM(const char* path, const LSTM* net, uint64_t episodes, uint64_t seed) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t inputSize = (uint32_t)net->inputSize;
    uint32_t hiddenSize = (uint32_t)net->hiddenSize;
    uint32_t outputSize = (uint32_t)net->outputSize;

    fwrite(CKPT_MAGIC, sizeof(CKPT_MAGIC), 1, f);
    fwrite(&CKPT_VERSION, sizeof(CKPT_VERSION), 1, f);
    fwrite(&inputSize, sizeof(inputSize), 1, f);
    fwrite(&hiddenSize, sizeof(hiddenSize), 1, f);
    fwrite(&outputSize, sizeof(outputSize), 1, f);
    fwrite(&episodes, sizeof(episodes), 1, f);
    fwrite(&seed, sizeof(seed), 1, f);

    int Z = (int)inputSize + (int)hiddenSize;

    for (int i=0; i<Z; i++) {
        fwrite(net->Wf[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wf_m[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wf_v[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wi[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wi_m[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wi_v[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wc[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wc_m[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wc_v[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wo[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wo_m[i], sizeof(double), hiddenSize, f);
        fwrite(net->Wo_v[i], sizeof(double), hiddenSize, f);
    }

    fwrite(net->Bf, sizeof(double), hiddenSize, f);
    fwrite(net->Bf_m, sizeof(double), hiddenSize, f);
    fwrite(net->Bf_v, sizeof(double), hiddenSize, f);
    fwrite(net->Bi, sizeof(double), hiddenSize, f);
    fwrite(net->Bi_m, sizeof(double), hiddenSize, f);
    fwrite(net->Bi_v, sizeof(double), hiddenSize, f);
    fwrite(net->Bc, sizeof(double), hiddenSize, f);
    fwrite(net->Bc_m, sizeof(double), hiddenSize, f);
    fwrite(net->Bc_v, sizeof(double), hiddenSize, f);
    fwrite(net->Bo, sizeof(double), hiddenSize, f);
    fwrite(net->Bo_m, sizeof(double), hiddenSize, f);
    fwrite(net->Bo_v, sizeof(double), hiddenSize, f);

    for (int i=0; i<(int)hiddenSize; i++) {
        fwrite(net->Wout[i], sizeof(double), outputSize, f);
        fwrite(net->Wout_m[i], sizeof(double), outputSize, f);
        fwrite(net->Wout_v[i], sizeof(double), outputSize, f);
    }

    fwrite(net->Bout, sizeof(double), outputSize, f);
    fwrite(net->Bout_m, sizeof(double), outputSize, f);
    fwrite(net->Bout_v, sizeof(double), outputSize, f);

    fwrite(net->hiddenState, sizeof(double), hiddenSize, f);
    fwrite(net->cellState, sizeof(double), hiddenSize, f);

    fwrite(&net->adam_t, sizeof(net->adam_t), 1, f);

    fwrite(net->Wv, sizeof(double), hiddenSize, f);
    fwrite(net->Wv_m, sizeof(double), hiddenSize, f);
    fwrite(net->Wv_v, sizeof(double), hiddenSize, f);

    fwrite(&net->Bv, sizeof(double), 1, f);
    fwrite(&net->Bv_m, sizeof(double), 1, f);
    fwrite(&net->Bv_v, sizeof(double), 1, f);

    fclose(f);
    return 0;
}

LSTM* loadLSTM(const char* path, uint64_t* episodes, uint64_t* seed) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[sizeof(CKPT_MAGIC)];
    uint32_t version;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint32_t outputSize;

    fread(magic, sizeof(magic), 1, f);
    fread(&version, sizeof(version), 1, f);
    fread(&inputSize, sizeof(inputSize), 1, f);
    fread(&hiddenSize, sizeof(hiddenSize), 1, f);
    fread(&outputSize, sizeof(outputSize), 1, f);
    fread(episodes, sizeof(*episodes), 1, f);
    fread(seed, sizeof(*seed), 1, f);

    LSTM* net = initLSTM((int)inputSize, (int)hiddenSize, (int)outputSize);

    int Z = (int)inputSize + (int)hiddenSize;

    for (int i=0; i<Z; i++) {
        fread(net->Wf[i], sizeof(double), hiddenSize, f);
        fread(net->Wf_m[i], sizeof(double), hiddenSize, f);
        fread(net->Wf_v[i], sizeof(double), hiddenSize, f);
        fread(net->Wi[i], sizeof(double), hiddenSize, f);
        fread(net->Wi_m[i], sizeof(double), hiddenSize, f);
        fread(net->Wi_v[i], sizeof(double), hiddenSize, f);
        fread(net->Wc[i], sizeof(double), hiddenSize, f);
        fread(net->Wc_m[i], sizeof(double), hiddenSize, f);
        fread(net->Wc_v[i], sizeof(double), hiddenSize, f);
        fread(net->Wo[i], sizeof(double), hiddenSize, f);
        fread(net->Wo_m[i], sizeof(double), hiddenSize, f);
        fread(net->Wo_v[i], sizeof(double), hiddenSize, f);
    }

    fread(net->Bf, sizeof(double), hiddenSize, f);
    fread(net->Bf_m, sizeof(double), hiddenSize, f);
    fread(net->Bf_v, sizeof(double), hiddenSize, f);
    fread(net->Bi, sizeof(double), hiddenSize, f);
    fread(net->Bi_m, sizeof(double), hiddenSize, f);
    fread(net->Bi_v, sizeof(double), hiddenSize, f);
    fread(net->Bc, sizeof(double), hiddenSize, f);
    fread(net->Bc_m, sizeof(double), hiddenSize, f);
    fread(net->Bc_v, sizeof(double), hiddenSize, f);
    fread(net->Bo, sizeof(double), hiddenSize, f);
    fread(net->Bo_m, sizeof(double), hiddenSize, f);
    fread(net->Bo_v, sizeof(double), hiddenSize, f);

    for (int i=0; i<(int)hiddenSize; i++) {
        fread(net->Wout[i], sizeof(double), outputSize, f);
        fread(net->Wout_m[i], sizeof(double), outputSize, f);
        fread(net->Wout_v[i], sizeof(double), outputSize, f);
    }

    fread(net->Bout, sizeof(double), outputSize, f);
    fread(net->Bout_m, sizeof(double), outputSize, f);
    fread(net->Bout_v, sizeof(double), outputSize, f);

    fread(net->hiddenState, sizeof(double), hiddenSize, f);
    fread(net->cellState, sizeof(double), hiddenSize, f);

    fread(&net->adam_t, sizeof(net->adam_t), 1, f);

    fread(net->Wv, sizeof(double), hiddenSize, f);
    fread(net->Wv_m, sizeof(double), hiddenSize, f);
    fread(net->Wv_v, sizeof(double), hiddenSize, f);

    fread(&net->Bv, sizeof(double), 1, f);
    fread(&net->Bv_m, sizeof(double), 1, f);
    fread(&net->Bv_v, sizeof(double), 1, f);

    fclose(f);
    return net;
}
