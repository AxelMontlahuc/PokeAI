#include "policy.h"

double xavierInitialization(double fanIn, double fanOut) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(6.0 / (fanIn + fanOut));
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
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
    assert(network->hiddenState != NULL && network->cellState != NULL && network->logits != NULL && network->probs != NULL);

    network->last_value = 0.0;
    network->adam_t = 0;

    int Z = inputSize + hiddenSize;

    network->Wf = malloc(Z * sizeof(double*));
    network->Wf_m = malloc(Z * sizeof(double*));
    network->Wf_v = malloc(Z * sizeof(double*));
    network->Wi = malloc(Z * sizeof(double*));
    network->Wi_m = malloc(Z * sizeof(double*));
    network->Wi_v = malloc(Z * sizeof(double*));
    network->Wc = malloc(Z * sizeof(double*));
    network->Wc_m = malloc(Z * sizeof(double*));
    network->Wc_v = malloc(Z * sizeof(double*));
    network->Wo = malloc(Z * sizeof(double*));
    network->Wo_m = malloc(Z * sizeof(double*));
    network->Wo_v = malloc(Z * sizeof(double*));
    assert(network->Wf != NULL && network->Wi != NULL && network->Wc != NULL && network->Wo != NULL && network->Wf_m && network->Wf_v && network->Wi_m && network->Wi_v && network->Wc_m && network->Wc_v && network->Wo_m && network->Wo_v);

    for (int i=0; i<Z; i++) {
        network->Wf[i] = malloc(hiddenSize * sizeof(double));
        network->Wf_m[i] = calloc(hiddenSize, sizeof(double));
        network->Wf_v[i] = calloc(hiddenSize, sizeof(double));
        network->Wi[i] = malloc(hiddenSize * sizeof(double));
        network->Wi_m[i] = calloc(hiddenSize, sizeof(double));
        network->Wi_v[i] = calloc(hiddenSize, sizeof(double));
        network->Wc[i] = malloc(hiddenSize * sizeof(double));
        network->Wc_m[i] = calloc(hiddenSize, sizeof(double));
        network->Wc_v[i] = calloc(hiddenSize, sizeof(double));
        network->Wo[i] = malloc(hiddenSize * sizeof(double));
        network->Wo_m[i] = calloc(hiddenSize, sizeof(double));
        network->Wo_v[i] = calloc(hiddenSize, sizeof(double));
        assert(network->Wf[i] != NULL && network->Wi[i] != NULL && network->Wc[i] != NULL && network->Wo[i] != NULL && network->Wf_m[i] && network->Wf_v[i] && network->Wi_m[i] && network->Wi_v[i] && network->Wc_m[i] && network->Wc_v[i] && network->Wo_m[i] && network->Wo_v[i]);

        for (int j=0; j<hiddenSize; j++) {
            network->Wf[i][j] = xavierInitialization(inputSize, hiddenSize);
            network->Wi[i][j] = xavierInitialization(inputSize, hiddenSize);
            network->Wc[i][j] = xavierInitialization(inputSize, hiddenSize);
            network->Wo[i][j] = xavierInitialization(inputSize, hiddenSize);
        }
    }

    network->Bf = calloc(hiddenSize, sizeof(double));
    network->Bf_m = calloc(hiddenSize, sizeof(double));
    network->Bf_v = calloc(hiddenSize, sizeof(double));
    network->Bi = calloc(hiddenSize, sizeof(double));
    network->Bi_m = calloc(hiddenSize, sizeof(double));
    network->Bi_v = calloc(hiddenSize, sizeof(double));
    network->Bc = calloc(hiddenSize, sizeof(double));
    network->Bc_m = calloc(hiddenSize, sizeof(double));
    network->Bc_v = calloc(hiddenSize, sizeof(double));
    network->Bo = calloc(hiddenSize, sizeof(double));
    network->Bo_m = calloc(hiddenSize, sizeof(double));
    network->Bo_v = calloc(hiddenSize, sizeof(double));
    assert(network->Bf != NULL && network->Bi != NULL && network->Bc != NULL && network->Bo != NULL && network->Bf_m != NULL && network->Bf_v != NULL && network->Bi_m != NULL && network->Bi_v != NULL && network->Bc_m != NULL && network->Bc_v != NULL && network->Bo_m != NULL && network->Bo_v != NULL);

    for (int i=0; i<hiddenSize; i++) network->Bf[i] = 1.0;

    network->Wout = malloc(hiddenSize * sizeof(double*));
    network->Wout_m = malloc(hiddenSize * sizeof(double*));
    network->Wout_v = malloc(hiddenSize * sizeof(double*));
    assert(network->Wout != NULL && network->Wout_m != NULL && network->Wout_v != NULL);

    for (int i=0; i<network->hiddenSize; i++) {
        network->Wout[i] = malloc(network->outputSize * sizeof(double));
        network->Wout_m[i] = calloc(network->outputSize, sizeof(double));
        network->Wout_v[i] = calloc(network->outputSize, sizeof(double));
        assert(network->Wout[i] != NULL && network->Wout_m[i] != NULL && network->Wout_v[i] != NULL);

        for (int j=0; j<network->outputSize; j++) {
            network->Wout[i][j] = xavierInitialization(network->hiddenSize, network->outputSize);
        }
    }

    network->Bout = calloc(network->outputSize, sizeof(double));
    network->Bout_m = calloc(network->outputSize, sizeof(double));
    network->Bout_v = calloc(network->outputSize, sizeof(double));
    assert(network->Bout != NULL && network->Bout_m != NULL && network->Bout_v != NULL);

    network->Wv = malloc(hiddenSize * sizeof(double));
    network->Wv_m = calloc(hiddenSize, sizeof(double));
    network->Wv_v = calloc(hiddenSize, sizeof(double));
    assert(network->Wv != NULL && network->Wv_m != NULL && network->Wv_v != NULL);

    for (int i = 0; i < hiddenSize; i++) network->Wv[i] = xavierInitialization(hiddenSize, 1);

    network->Bv = 0.0;
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
        free(network->Wf_m[i]);
        free(network->Wf_v[i]);
        free(network->Wi[i]);
        free(network->Wi_m[i]);
        free(network->Wi_v[i]);
        free(network->Wc[i]);
        free(network->Wc_m[i]);
        free(network->Wc_v[i]);
        free(network->Wo[i]);
        free(network->Wo_m[i]);
        free(network->Wo_v[i]);
    }

    free(network->Wf);
    free(network->Wf_m);
    free(network->Wf_v);
    free(network->Wi);
    free(network->Wi_m);
    free(network->Wi_v);
    free(network->Wc);
    free(network->Wc_m);
    free(network->Wc_v);
    free(network->Wo);
    free(network->Wo_m);
    free(network->Wo_v);

    free(network->Bf);
    free(network->Bf_m);
    free(network->Bf_v);
    free(network->Bi);
    free(network->Bi_m);
    free(network->Bi_v);
    free(network->Bc);
    free(network->Bc_m);
    free(network->Bc_v);
    free(network->Bo);
    free(network->Bo_m);
    free(network->Bo_v);

    for (int i=0; i<network->hiddenSize; i++) {
        free(network->Wout[i]);
        free(network->Wout_m[i]);
        free(network->Wout_v[i]);
    }

    free(network->Wout);
    free(network->Wout_m);
    free(network->Wout_v);

    free(network->Bout);
    free(network->Bout_m);
    free(network->Bout_v);

    free(network->Wv);
    free(network->Wv_m);
    free(network->Wv_v);

    free(network);
}

double* forgetGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bf[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wf[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* inputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bi[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wi[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* cellGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bc[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wc[j][i];
        }
        result[i] = tanh(result[i]);
    }

    return result;
}

double* outputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bo[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wo[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* forward(LSTM* network, double* data, double temperature) {
    double* combinedState = malloc((network->hiddenSize + network->inputSize) * sizeof(double));
    assert(combinedState != NULL);

    for (int i = 0; i < network->inputSize; i++) combinedState[i] = data[i];
    for (int i = 0; i < network->hiddenSize; i++) combinedState[i + network->inputSize] = network->hiddenState[i];

    double* fArray = forgetGate(network, combinedState);
    double* iArray = inputGate(network, combinedState);
    double* cArray = cellGate(network, combinedState);
    double* oArray = outputGate(network, combinedState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * cArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    for (int i=0; i<network->outputSize; i++) {
        double s = network->Bout[i];
        for (int j=0; j<network->hiddenSize; j++) s += network->hiddenState[j] * network->Wout[j][i];
        network->logits[i] = s;
    }

    double v = network->Bv;
    for (int i = 0; i < network->hiddenSize; i++) v += network->hiddenState[i] * network->Wv[i];
    network->last_value = v;

    double maxlog = network->logits[0];
    for (int k = 1; k < network->outputSize; k++) {
        if (network->logits[k] > maxlog) maxlog = network->logits[k];
    }

    double sum = 0.0;
    for (int k = 0; k < network->outputSize; k++) {
        double e = exp((network->logits[k] - maxlog) / temperature);
        network->probs[k] = e;
        sum += e;
    }
    for (int k = 0; k < network->outputSize; k++) network->probs[k] /= sum + NUM_EPS;

    free(combinedState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);

    return network->probs;
}

void entropyBonus(const double* probs, int O, double coeff, double* dlogits) {
    double totalH = 0.0; // En fait c'est -H

    for (int k = 0; k < O; k++) {
        totalH += probs[k] * log(probs[k] + NUM_EPS);
    }
    
    for (int k = 0; k < O; k++) {
        double gradH = probs[k] * (totalH - log(probs[k] + NUM_EPS));
        dlogits[k] += coeff * gradH;
    }
}

void dL_dWout(LSTM* network, double* dlogits, double* h_t, double** dWout, double* dBout, double* dh_accum) {
    for (int j = 0; j < network->hiddenSize; j++) {
        for (int k = 0; k < network->outputSize; k++) dWout[j][k] += h_t[j] * dlogits[k];
    }
    for (int k = 0; k < network->outputSize; k++) dBout[k] += dlogits[k];
    for (int j = 0; j < network->hiddenSize; j++) {
        double s = 0.0;
        for (int k = 0; k < network->outputSize; k++) s += network->Wout[j][k] * dlogits[k];
        dh_accum[j] += s;
    }
}

static inline void adam_update_scalar(double* param, double* m, double* v, double grad, double lr, double beta1, double beta2, double inv_bc1, double inv_bc2, double eps, double scale) {
    double g = grad * scale;
    *m = beta1 * (*m) + (1.0 - beta1) * g;
    *v = beta2 * (*v) + (1.0 - beta2) * (g * g);
    double mhat = (*m) * inv_bc1;
    double vhat = (*v) * inv_bc2;
    *param += lr * (mhat / (sqrt(vhat) + eps));
}

static void adam_update_vector(double* P, double* M, double* V, const double* G, int n, double lr, double beta1, double beta2, double inv_bc1, double inv_bc2, double eps, double scale) {
    for (int i = 0; i < n; i++) {
        adam_update_scalar(&P[i], &M[i], &V[i], G[i], lr, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    }
}

static void adam_update_matrix(double** P, double** M, double** V, double** G, int rows, int cols, double lr, double beta1, double beta2, double inv_bc1, double inv_bc2, double eps, double scale) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            adam_update_scalar(&P[r][c], &M[r][c], &V[r][c], G[r][c], lr, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
        }
    }
}

static inline double clip_grad(double ratio, double advantage, double clip_eps) {
    double low = 1.0 - clip_eps;
    double high = 1.0 + clip_eps;
    bool clipped = (advantage > 0.0 && ratio > high) || (advantage < 0.0 && ratio < low);
    return clipped ? 0.0 : (ratio * advantage);
}

void backpropagation(LSTM* network, double learningRate, int steps, trajectory** trajectories, int batch_count, double temperature, BackpropStats* stats) {
    const int T = steps;
    const int I = network->inputSize;
    const int H = network->hiddenSize;
    const int O = network->outputSize;
    const int Z = I + H;

    double** dWf = malloc(Z * sizeof(double*));
    double** dWi = malloc(Z * sizeof(double*));
    double** dWc = malloc(Z * sizeof(double*));
    double** dWo = malloc(Z * sizeof(double*));

    for (int i = 0; i < Z; i++) {
        dWf[i] = calloc(H, sizeof(double));
        dWi[i] = calloc(H, sizeof(double));
        dWc[i] = calloc(H, sizeof(double));
        dWo[i] = calloc(H, sizeof(double));
        assert(dWf[i] != NULL && dWi[i] != NULL && dWc[i] != NULL && dWo[i] != NULL);
    }

    double** dWout = malloc(H * sizeof(double*));
    for (int j = 0; j < H; j++) {
        dWout[j] = calloc(O, sizeof(double));
        assert(dWout[j] != NULL);
    }

    double* dBf = calloc(H, sizeof(double));
    double* dBi = calloc(H, sizeof(double));
    double* dBc = calloc(H, sizeof(double));
    double* dBo = calloc(H, sizeof(double));
    double* dBout = calloc(O, sizeof(double));
    double* dWv = calloc(H, sizeof(double));
    double dBv = 0.0;
    assert(dBf != NULL && dBi != NULL && dBc != NULL && dBo != NULL && dBout != NULL && dWv != NULL);

    double** As = malloc(batch_count * sizeof(double*));
    double** Rs = malloc(batch_count * sizeof(double*));
    double* allA = malloc((size_t)batch_count * T * sizeof(double));
    assert(As != NULL && Rs != NULL && allA != NULL);

    int cursor = 0;
    for (int b = 0; b < batch_count; b++) {
        As[b] = malloc((size_t)T * sizeof(double));
        Rs[b] = malloc((size_t)T * sizeof(double));
        assert(As[b] != NULL && Rs[b] != NULL);

        computeGAE(trajectories[b]->rewards, trajectories[b]->values, T, GAMMA_DISCOUNT, GAE_LAMBDA, As[b], Rs[b]);
        for (int t = 0; t < T; t++) allA[cursor++] = As[b][t];
    }
    normPNL(allA, cursor);
    
    cursor = 0;
    for (int b = 0; b < batch_count; b++) {
        for (int t = 0; t < T; t++) As[b][t] = allA[cursor++];
    }

    double** x = malloc(T * sizeof(double*));
    double** z = malloc(T * sizeof(double*));
    double** f = malloc(T * sizeof(double*));
    double** i = malloc(T * sizeof(double*));
    double** g = malloc(T * sizeof(double*));
    double** o = malloc(T * sizeof(double*));
    double** c = malloc(T * sizeof(double*));
    double** h = malloc(T * sizeof(double*));
    double** cprev = malloc(T * sizeof(double*));
    double** tanhC = malloc(T * sizeof(double*));
    assert(x != NULL && z != NULL && f != NULL && i != NULL && g != NULL && o != NULL && c != NULL && h != NULL && cprev != NULL && tanhC != NULL);

    double* x_block = malloc((size_t)T * I * sizeof(double));
    double* z_block = malloc((size_t)T * Z * sizeof(double));
    double* f_block = malloc((size_t)T * H * sizeof(double));
    double* i_block = malloc((size_t)T * H * sizeof(double));
    double* g_block = malloc((size_t)T * H * sizeof(double));
    double* o_block = malloc((size_t)T * H * sizeof(double));
    double* c_block = malloc((size_t)T * H * sizeof(double));
    double* h_block = malloc((size_t)T * H * sizeof(double));
    double* cprev_block = malloc((size_t)T * H * sizeof(double));
    double* tanhC_block = malloc((size_t)T * H * sizeof(double));
    assert(x_block != NULL && z_block != NULL && f_block != NULL && i_block != NULL && g_block != NULL && o_block != NULL && c_block != NULL && h_block != NULL && cprev_block != NULL && tanhC_block != NULL);

    for (int t = 0; t < T; t++) {
        x[t] = &x_block[(size_t)t * I];
        z[t] = &z_block[(size_t)t * Z];
        f[t] = &f_block[(size_t)t * H];
        i[t] = &i_block[(size_t)t * H];
        g[t] = &g_block[(size_t)t * H];
        o[t] = &o_block[(size_t)t * H];
        c[t] = &c_block[(size_t)t * H];
        h[t] = &h_block[(size_t)t * H];
        cprev[t] = &cprev_block[(size_t)t * H];
        tanhC[t] = &tanhC_block[(size_t)t * H];
    }

    double* hprev = malloc(H * sizeof(double));
    double* cprev_vec = malloc(H * sizeof(double));
    double* dh = malloc(H * sizeof(double));
    double* dh_next = malloc(H * sizeof(double));
    double* dc = malloc(H * sizeof(double));
    double* dc_next = malloc(H * sizeof(double));
    double* df = malloc(H * sizeof(double));
    double* d_f_pre = malloc(H * sizeof(double));
    double* di_vec = malloc(H * sizeof(double));
    double* d_i_pre = malloc(H * sizeof(double));
    double* dg = malloc(H * sizeof(double));
    double* d_g_pre = malloc(H * sizeof(double));
    double* do_vec = malloc(H * sizeof(double));
    double* d_o_pre = malloc(H * sizeof(double));
    double* dlogits = malloc(O * sizeof(double));
    double* logits_now = malloc(O * sizeof(double));
    double* probs_now = malloc(O * sizeof(double));
    double* dz = malloc(Z * sizeof(double));
    double* sf = malloc(H * sizeof(double));
    double* si = malloc(H * sizeof(double));
    double* sg = malloc(H * sizeof(double));
    double* so = malloc(H * sizeof(double));
    assert(hprev != NULL && cprev_vec != NULL && dh != NULL && dh_next != NULL && dc != NULL && dc_next != NULL && df != NULL && d_f_pre != NULL && di_vec != NULL && d_i_pre != NULL && dg != NULL && d_g_pre != NULL && do_vec != NULL && d_o_pre != NULL && sf != NULL && si != NULL && sg != NULL && so != NULL);
    assert(dlogits != NULL && logits_now != NULL && probs_now != NULL);
    assert(dz != NULL);

    for (int b = 0; b < batch_count; b++) {
        trajectory* tr = trajectories[b];

        for (int j = 0; j < H; j++) { 
            hprev[j] = 0.0; 
            cprev_vec[j] = 0.0; 
        }

        for (int t = 0; t < T; t++) {
            convertState(tr->states[t], x[t]);
            memcpy(z[t], x[t], (size_t)I * sizeof(double));
            for (int a = 0; a < H; a++) z[t][I + a] = hprev[a];
            for (int j = 0; j < H; j++) cprev[t][j] = cprev_vec[j];

            for (int j = 0; j < H; j++) { 
                sf[j] = network->Bf[j]; 
                si[j] = network->Bi[j]; 
                sg[j] = network->Bc[j]; 
                so[j] = network->Bo[j]; 
            }

            for (int a = 0; a < Z; a++) {
                double v = z[t][a];
                double* Wfr = network->Wf[a];
                double* Wir = network->Wi[a];
                double* Wcr = network->Wc[a];
                double* Wor = network->Wo[a];
                for (int j = 0; j < H; j++) {
                    sf[j] += v * Wfr[j];
                    si[j] += v * Wir[j];
                    sg[j] += v * Wcr[j];
                    so[j] += v * Wor[j];
                }
            }

            for (int j = 0; j < H; j++) {
                double fj = sigmoid(sf[j]);
                double ij = sigmoid(si[j]);
                double gj = tanh(sg[j]);
                double oj = sigmoid(so[j]);
                f[t][j] = fj; i[t][j] = ij; g[t][j] = gj; o[t][j] = oj;
                c[t][j] = fj * cprev_vec[j] + ij * gj;
                double tc = tanh(c[t][j]);
                tanhC[t][j] = tc;
                h[t][j] = oj * tc;
                hprev[j] = h[t][j];
                cprev_vec[j] = c[t][j];
            }
        }

        for (int j = 0; j < H; j++) { 
            dh_next[j] = 0.0; 
            dc_next[j] = 0.0; 
        }
        for (int t = T - 1; t >= 0; t--) {
            int actionIndex = tr->actions[t];
            double maxlog = -1e30;

            for (int k = 0; k < O; k++) {
                double s = network->Bout[k];
                for (int j = 0; j < H; j++) s += h[t][j] * network->Wout[j][k];
                logits_now[k] = s;
                if (s > maxlog) maxlog = s;
            }

            double sum = 0.0;
            for (int k = 0; k < O; k++) {
                double e = exp((logits_now[k] - maxlog) / temperature);
                probs_now[k] = e;
                sum += e;
            }
            for (int k = 0; k < O; k++) probs_now[k] /= (sum + NUM_EPS);

            double logp_new = log(probs_now[actionIndex] + NUM_EPS);
            double logp_old = log(tr->probs[t][actionIndex] + NUM_EPS);
            double ratio = exp(logp_new - logp_old);
            double A = As[b][t];
            double grad_weight = clip_grad(ratio, A, CLIP_EPS);

            for (int k = 0; k < O; k++) {
                double base = ((k == actionIndex) ? 1.0 : 0.0) - probs_now[k];
                dlogits[k] = base * grad_weight;
            }

            entropyBonus(probs_now, O, ENTROPY_COEFF, dlogits);

            double inv_temp = 1.0 / temperature;
            for (int k = 0; k < O; k++) dlogits[k] *= inv_temp;

            for (int j = 0; j < H; j++) dh[j] = dh_next[j];
            dL_dWout(network, dlogits, h[t], dWout, dBout, dh);

            double v_t = network->Bv;
            for (int j = 0; j < H; j++) v_t += h[t][j] * network->Wv[j];
            double v_old = tr->values[t];
            double R_t = Rs[b][t];
            double grad_v;
            if (VALUE_CLIP_EPS > 0.0) {
                double v_low = v_old - VALUE_CLIP_EPS;
                double v_high = v_old + VALUE_CLIP_EPS;
                double v_t_clipped = fmin(fmax(v_t, v_low), v_high);
                double un = (v_t - R_t) * (v_t - R_t);
                double cl = (v_t_clipped - R_t) * (v_t_clipped - R_t);
                if (un >= cl) {
                    grad_v = -2.0 * VALUE_COEFF * (v_t - R_t);
                } else {
                    double dvclip_dv = (v_t > v_high || v_t < v_low) ? 0.0 : 1.0;
                    grad_v = -2.0 * VALUE_COEFF * (v_t_clipped - R_t) * dvclip_dv;
                }
            } else {
                grad_v = -2.0 * VALUE_COEFF * (v_t - R_t);
            }
            for (int j = 0; j < H; j++) dWv[j] += grad_v * h[t][j];
            dBv += grad_v;
            for (int j = 0; j < H; j++) dh[j] += grad_v * network->Wv[j];

            for (int j = 0; j < H; j++) do_vec[j] = dh[j] * tanhC[t][j];
            for (int j = 0; j < H; j++) d_o_pre[j] = do_vec[j] * o[t][j] * (1.0 - o[t][j]);
            for (int j = 0; j < H; j++) {
                double tc = tanhC[t][j];
                dc[j] = dh[j] * o[t][j] * (1.0 - tc * tc) + dc_next[j];
            }
            for (int j = 0; j < H; j++) df[j] = dc[j] * cprev[t][j];
            for (int j = 0; j < H; j++) di_vec[j] = dc[j] * g[t][j];
            for (int j = 0; j < H; j++) dg[j] = dc[j] * i[t][j];
            for (int j = 0; j < H; j++) d_f_pre[j] = df[j] * f[t][j] * (1.0 - f[t][j]);
            for (int j = 0; j < H; j++) d_i_pre[j] = di_vec[j] * i[t][j] * (1.0 - i[t][j]);
            for (int j = 0; j < H; j++) d_g_pre[j] = dg[j] * (1.0 - g[t][j] * g[t][j]);

            for (int j = 0; j < H; j++) { 
                dBf[j] += d_f_pre[j]; 
                dBi[j] += d_i_pre[j]; 
                dBc[j] += d_g_pre[j]; 
                dBo[j] += d_o_pre[j]; 
            }

            for (int a = 0; a < Z; a++) {
                double vz = z[t][a];
                for (int j = 0; j < H; j++) {
                    dWf[a][j] += vz * d_f_pre[j];
                    dWi[a][j] += vz * d_i_pre[j];
                    dWc[a][j] += vz * d_g_pre[j];
                    dWo[a][j] += vz * d_o_pre[j];
                }
            }

            for (int a = 0; a < Z; a++) {
                double s = 0.0;
                const double* Wfr = network->Wf[a];
                const double* Wir = network->Wi[a];
                const double* Wcr = network->Wc[a];
                const double* Wor = network->Wo[a];
                for (int j = 0; j < H; j++) {
                    s += Wfr[j] * d_f_pre[j];
                    s += Wir[j] * d_i_pre[j];
                    s += Wcr[j] * d_g_pre[j];
                    s += Wor[j] * d_o_pre[j];
                }
                dz[a] = s;
            }

            for (int j = 0; j < H; j++) dh_next[j] = dz[I + j];
            for (int j = 0; j < H; j++) dc_next[j] = dc[j] * f[t][j];
        }
    }

    double invM = 1.0 / (double)batch_count;
    for (int a = 0; a < Z; a++) {
        for (int j = 0; j < H; j++) { 
            dWf[a][j] *= invM; 
            dWi[a][j] *= invM; 
            dWc[a][j] *= invM; 
            dWo[a][j] *= invM; 
        } 
    }

    for (int j = 0; j < H; j++) { 
        for (int k = 0; k < O; k++) dWout[j][k] *= invM; 
    }

    for (int j = 0; j < H; j++) { 
        dBf[j] *= invM; 
        dBi[j] *= invM; 
        dBc[j] *= invM; 
        dBo[j] *= invM; 
    }

    for (int k = 0; k < O; k++) dBout[k] *= invM;
    for (int j = 0; j < H; j++) dWv[j] *= invM;
    dBv *= invM;

    double clip = GRAD_CLIP_NORM;
    double norm2 = 0.0;
    for (int a = 0; a < Z; a++) { 
        for (int j = 0; j < H; j++) { 
            norm2 += dWf[a][j]*dWf[a][j]; 
            norm2 += dWi[a][j]*dWi[a][j]; 
            norm2 += dWc[a][j]*dWc[a][j]; 
            norm2 += dWo[a][j]*dWo[a][j]; 
        } 
    }
    for (int j = 0; j < H; j++) { 
        for (int k = 0; k < O; k++) norm2 += dWout[j][k]*dWout[j][k]; 
    }
    for (int j = 0; j < H; j++) norm2 += dBf[j]*dBf[j] + dBi[j]*dBi[j] + dBc[j]*dBc[j] + dBo[j]*dBo[j];
    for (int k = 0; k < O; k++) norm2 += dBout[k]*dBout[k];
    for (int j = 0; j < H; j++) norm2 += dWv[j]*dWv[j];
    norm2 += dBv * dBv;
    double norm = sqrt(norm2);
    double scale = (norm > clip) ? (clip / (norm + NUM_EPS)) : 1.0;
    if (stats) {
        stats->grad_norm = norm;
        stats->clip_scale = scale;
    }

    const double beta1 = ADAM_BETA1;
    const double beta2 = ADAM_BETA2;
    const double eps = ADAM_EPS;
    network->adam_t += 1;
    double bc1 = 1.0 - pow(beta1, (double)network->adam_t);
    double bc2 = 1.0 - pow(beta2, (double)network->adam_t);
    double inv_bc1 = 1.0 / bc1;
    double inv_bc2 = 1.0 / bc2;

    adam_update_matrix(network->Wf, network->Wf_m, network->Wf_v, dWf, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_matrix(network->Wi, network->Wi_m, network->Wi_v, dWi, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_matrix(network->Wc, network->Wc_m, network->Wc_v, dWc, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_matrix(network->Wo, network->Wo_m, network->Wo_v, dWo, Z, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_matrix(network->Wout, network->Wout_m, network->Wout_v, dWout, H, O, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);

    adam_update_vector(network->Bf, network->Bf_m, network->Bf_v, dBf, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_vector(network->Bi, network->Bi_m, network->Bi_v, dBi, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_vector(network->Bc, network->Bc_m, network->Bc_v, dBc, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_vector(network->Bo, network->Bo_m, network->Bo_v, dBo, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_vector(network->Bout, network->Bout_m, network->Bout_v, dBout, O, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);

    adam_update_vector(network->Wv, network->Wv_m, network->Wv_v, dWv, H, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);
    adam_update_scalar(&network->Bv, &network->Bv_m, &network->Bv_v, dBv, learningRate, beta1, beta2, inv_bc1, inv_bc2, eps, scale);

    for (int a = 0; a < Z; a++) { 
        free(dWf[a]); 
        free(dWi[a]); 
        free(dWc[a]); 
        free(dWo[a]); 
    }
    for (int j = 0; j < H; j++) free(dWout[j]);
    free(dWf); 
    free(dWi); 
    free(dWc); 
    free(dWo); 
    free(dWout);
    free(dBf);
    free(dBi);
    free(dBc);
    free(dBo);
    free(dBout);
    free(dWv);

    for (int b = 0; b < batch_count; b++) { 
        free(As[b]); 
        free(Rs[b]); 
    }
    free(As); 
    free(Rs); 
    free(allA);

    free(x); 
    free(z); 
    free(f); 
    free(i); 
    free(g); 
    free(o); 
    free(c); 
    free(h); 
    free(cprev); 
    free(tanhC);
    free(x_block); 
    free(z_block); 
    free(f_block); 
    free(i_block); 
    free(g_block); 
    free(o_block); 
    free(c_block); 
    free(h_block); 
    free(cprev_block); 
    free(tanhC_block);

    free(hprev); 
    free(cprev_vec); 
    free(dh); 
    free(dh_next); 
    free(dc); 
    free(dc_next); 
    free(df); 
    free(d_f_pre); 
    free(di_vec); 
    free(d_i_pre); 
    free(dg); 
    free(d_g_pre); 
    free(do_vec); 
    free(d_o_pre); 
    free(sf); 
    free(si); 
    free(sg); 
    free(so);

    free(dlogits); 
    free(logits_now); 
    free(probs_now);

    free(dz);
}