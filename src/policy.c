#include "policy.h"
#include "state.h"

static const double ENTROPY_COEFF = 0.05;

void entropyBonus(const double* probs, int O, double coeff, double* dlogits) {
    double mean_logp = 0.0;
    for (int k = 0; k < O; k++) {
        double pk = probs[k];
        mean_logp += pk * log(pk + 1e-12);
    }
    for (int k = 0; k < O; k++) {
        double pk = probs[k];
        double ent_grad = pk * (mean_logp - log(pk + 1e-12));
        dlogits[k] += coeff * ent_grad;
    }
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

    for (int i = 0; i < network->inputSize; i++) {
        combinedState[i] = data[i];
    }
    for (int i = 0; i < network->hiddenSize; i++) combinedState[i + network->inputSize] = network->hiddenState[i];

    double* fArray = forgetGate(network, combinedState);
    double* iArray = inputGate(network, combinedState);
    double* cArray = cellGate(network, combinedState);
    double* oArray = outputGate(network, combinedState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * cArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    for (int k=0; k<network->outputSize; k++) {
        double s = network->Bout[k];
        for (int j=0; j<network->hiddenSize; j++) s += network->hiddenState[j] * network->Wout[j][k];
        network->logits[k] = s;
    }

    double v = network->Bv;
    for (int j = 0; j < network->hiddenSize; j++) v += network->hiddenState[j] * network->Wv[j];
    network->last_value = v;

    int O = network->outputSize;
    double maxlog = network->logits[0];
    for (int k = 1; k < O; k++) {
        if (network->logits[k] > maxlog) maxlog = network->logits[k];
    }

    double sum = 0.0;
    for (int k = 0; k < O; k++) {
        double z = (network->logits[k] - maxlog) / temperature;
        double e = exp(z);
        network->probs[k] = e;
        sum += e;
    }
    double inv = 1.0 / (sum + 1e-12);
    for (int k = 0; k < O; k++) network->probs[k] *= inv;

    free(combinedState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);

    return network->probs;
}

void dL_dWout(LSTM* network, double* dlogits, double* h_t, double** dWout, double* dBout, double* dh_accum) {
    int H = network->hiddenSize;
    int O = network->outputSize;
    for (int j = 0; j < H; j++) {
        for (int k = 0; k < O; k++) dWout[j][k] += h_t[j] * dlogits[k];
    }
    for (int k = 0; k < O; k++) dBout[k] += dlogits[k];
    for (int j = 0; j < H; j++) {
        double s = 0.0;
        for (int k = 0; k < O; k++) s += network->Wout[j][k] * dlogits[k];
        dh_accum[j] += s;
    }
}

static int actionToIndex(MGBAButton action) {
    switch (action) {
        case MGBA_BUTTON_UP: return 0;
        case MGBA_BUTTON_DOWN: return 1;
        case MGBA_BUTTON_LEFT: return 2;
        case MGBA_BUTTON_RIGHT: return 3;
        case MGBA_BUTTON_A: return 4;
        case MGBA_BUTTON_B: return 5;
        case MGBA_BUTTON_START: return 6;
        default: return 5;
    }
}


void backpropagation(LSTM* network, double learningRate, int steps, trajectory** trajectories, int batchCount, double temperature, double epsilon, BackpropStats* stats) {
    int H = network->hiddenSize;
    int I = network->inputSize;
    int Z = I + H;
    int O = network->outputSize;

    int T = steps;
    int totalSteps = batchCount * steps;
    double** Gs = malloc(batchCount * sizeof(double*));
    double* allG = malloc(totalSteps * sizeof(double));
    assert(Gs != NULL && allG != NULL);

    int cur = 0;
    for (int b = 0; b < batchCount; b++) {
        Gs[b] = discountedPNL(trajectories[b]->rewards, 0.9, T);
        for (int t = 0; t < T; t++) allG[cur++] = Gs[b][t];
    }

    normPNL(allG, totalSteps);

    cur = 0;
    for (int b = 0; b < batchCount; b++) {
        for (int t = 0; t < T; t++) Gs[b][t] = allG[cur++];
    }

    double** dWf = malloc(Z * sizeof(double*));
    double** dWi = malloc(Z * sizeof(double*));
    double** dWc = malloc(Z * sizeof(double*));
    double** dWo = malloc(Z * sizeof(double*));
    double** dWout = malloc(H * sizeof(double*));
    assert(dWf && dWi && dWc && dWo && dWout);

    for (int a = 0; a < Z; a++) {
        dWf[a] = calloc(H, sizeof(double));
        dWi[a] = calloc(H, sizeof(double));
        dWc[a] = calloc(H, sizeof(double));
        dWo[a] = calloc(H, sizeof(double));
        assert(dWf[a] && dWi[a] && dWc[a] && dWo[a]);
    }

    for (int j = 0; j < H; j++) { 
        dWout[j] = calloc(O, sizeof(double)); 
        assert(dWout[j]); 
    }

    double* dBf = calloc(H, sizeof(double));
    double* dBi = calloc(H, sizeof(double));
    double* dBc = calloc(H, sizeof(double));
    double* dBo = calloc(H, sizeof(double));
    double* dBout = calloc(O, sizeof(double));
    assert(dBf && dBi && dBc && dBo && dBout);

    for (int b = 0; b < batchCount; b++) {
        trajectory* tr = trajectories[b];
        
        double** x = malloc(T * sizeof(double*));
        double** z = malloc(T * sizeof(double*));
        double** f = malloc(T * sizeof(double*));
        double** i = malloc(T * sizeof(double*));
        double** g = malloc(T * sizeof(double*));
        double** o = malloc(T * sizeof(double*));
        double** c = malloc(T * sizeof(double*));
        double** h = malloc(T * sizeof(double*));
        double** cprev = malloc(T * sizeof(double*));
        assert(x && z && f && i && g && o && c && h && cprev);

        double* hprev = calloc(H, sizeof(double));
        double* cprev_vec = calloc(H, sizeof(double));
        assert(hprev && cprev_vec);

        for (int t = 0; t < T; t++) {
            x[t] = convertState(tr->states[t]);
            z[t] = malloc(Z * sizeof(double));
            assert(z[t] != NULL);
            for (int a = 0; a < I; a++) z[t][a] = x[t][a];
            for (int a = 0; a < H; a++) z[t][I + a] = hprev[a];

            f[t] = malloc(H * sizeof(double));
            i[t] = malloc(H * sizeof(double));
            g[t] = malloc(H * sizeof(double));
            o[t] = malloc(H * sizeof(double));
            c[t] = malloc(H * sizeof(double));
            h[t] = malloc(H * sizeof(double));
            cprev[t] = malloc(H * sizeof(double));
            assert(f[t] != NULL && i[t] != NULL && g[t] != NULL && o[t] != NULL && c[t] != NULL && h[t] != NULL && cprev[t] != NULL);

            for (int j = 0; j < H; j++) cprev[t][j] = cprev_vec[j];

            for (int j = 0; j < H; j++) {
                double s = network->Bf[j];
                for (int a = 0; a < Z; a++) s += z[t][a] * network->Wf[a][j];
                f[t][j] = sigmoid(s);
            }
            for (int j = 0; j < H; j++) {
                double s = network->Bi[j];
                for (int a = 0; a < Z; a++) s += z[t][a] * network->Wi[a][j];
                i[t][j] = sigmoid(s);
            }
            for (int j = 0; j < H; j++) {
                double s = network->Bc[j];
                for (int a = 0; a < Z; a++) s += z[t][a] * network->Wc[a][j];
                g[t][j] = tanh(s);
            }
            for (int j = 0; j < H; j++) {
                double s = network->Bo[j];
                for (int a = 0; a < Z; a++) s += z[t][a] * network->Wo[a][j];
                o[t][j] = sigmoid(s);
            }
            for (int j = 0; j < H; j++) c[t][j] = f[t][j] * cprev_vec[j] + i[t][j] * g[t][j];
            for (int j = 0; j < H; j++) h[t][j] = o[t][j] * tanh(c[t][j]);

            for (int j = 0; j < H; j++) hprev[j] = h[t][j];
            for (int j = 0; j < H; j++) cprev_vec[j] = c[t][j];
        }

        double* G = Gs[b];

        double* dh_next = calloc(H, sizeof(double));
        double* dc_next = calloc(H, sizeof(double));
        assert(dh_next && dc_next);

        double* dlogits = malloc(O * sizeof(double));
        double* dh = malloc(H * sizeof(double));
        double* do_vec = malloc(H * sizeof(double));
        double* d_o_pre = malloc(H * sizeof(double));
        double* dc = malloc(H * sizeof(double));
        double* df = malloc(H * sizeof(double));
        double* di_vec = malloc(H * sizeof(double));
        double* dg = malloc(H * sizeof(double));
        double* d_f_pre = malloc(H * sizeof(double));
        double* d_i_pre = malloc(H * sizeof(double));
        double* d_g_pre = malloc(H * sizeof(double));
        double* dz = malloc(Z * sizeof(double));
        assert(dlogits && dh && do_vec && d_o_pre && dc && df && di_vec && dg && d_f_pre && d_i_pre && d_g_pre && dz);

        for (int t = T - 1; t >= 0; t--) {
            int actionIndex = actionToIndex(tr->actions[t]);
            for (int k = 0; k < O; k++) {
                double gadv = ((k == actionIndex) ? 1.0 : 0.0) - tr->probs[t][k];
                dlogits[k] = gadv * G[t];
            }

            entropyBonus(tr->probs[t], O, ENTROPY_COEFF, dlogits);

            double inv_temp = 1.0 / temperature;
            double scale_eps = fmax(0.0, 1.0 - epsilon);
            for (int k = 0; k < O; k++) dlogits[k] *= (inv_temp * scale_eps);

            for (int j = 0; j < H; j++) dh[j] = dh_next[j];
            dL_dWout(network, dlogits, h[t], dWout, dBout, dh);

            for (int j = 0; j < H; j++) do_vec[j] = dh[j] * tanh(c[t][j]);
            for (int j = 0; j < H; j++) d_o_pre[j] = do_vec[j] * o[t][j] * (1.0 - o[t][j]);
            for (int j = 0; j < H; j++) dc[j] = dh[j] * o[t][j] * (1.0 - tanh(c[t][j]) * tanh(c[t][j])) + dc_next[j];
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
                for (int j = 0; j < H; j++) {
                    dWf[a][j] += z[t][a] * d_f_pre[j];
                    dWi[a][j] += z[t][a] * d_i_pre[j];
                    dWc[a][j] += z[t][a] * d_g_pre[j];
                    dWo[a][j] += z[t][a] * d_o_pre[j];
                } 
            }

            for (int a = 0; a < Z; a++) {
                double s = 0.0;
                for (int j = 0; j < H; j++) {
                    s += network->Wf[a][j] * d_f_pre[j];
                    s += network->Wi[a][j] * d_i_pre[j];
                    s += network->Wc[a][j] * d_g_pre[j];
                    s += network->Wo[a][j] * d_o_pre[j];
                }
                dz[a] = s;
            }

            for (int j = 0; j < H; j++) dh_next[j] = dz[I + j];
            for (int j = 0; j < H; j++) dc_next[j] = dc[j] * f[t][j];

        }

        free(dlogits);
        free(dh);
        free(do_vec);
        free(d_o_pre);
        free(dc);
        free(df);
        free(di_vec);
        free(dg);
        free(d_f_pre);
        free(d_i_pre);
        free(d_g_pre);
        free(dz);

        for (int t = 0; t < T; t++) {
            free(x[t]);
            free(z[t]);
            free(f[t]);
            free(i[t]);
            free(g[t]);
            free(o[t]);
            free(c[t]);
            free(h[t]);
            free(cprev[t]);
        }

        free(x);
        free(z);
        free(f);
        free(i);
        free(g);
        free(o);
        free(c);
        free(h);
        free(cprev);
        free(hprev);
        free(cprev_vec);
        free(dh_next);
        free(dc_next);
    }

    double invM = 1.0 / (double)batchCount;
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

    double clip = 1.0; 
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

    double norm = sqrt(norm2); 
    double scale = (norm > clip) ? (clip / (norm + 1e-12)) : 1.0;
    if (stats) {
        stats->grad_norm = norm;
        stats->clip_scale = scale;
    }

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    network->adam_t += 1;
    double bc1 = 1.0 - pow(beta1, (double)network->adam_t);
    double bc2 = 1.0 - pow(beta2, (double)network->adam_t);
    double inv_bc1 = 1.0 / bc1;
    double inv_bc2 = 1.0 / bc2;

    for (int a = 0; a < Z; a++) {
        for (int j = 0; j < H; j++) {
            double g;
            
            g = dWf[a][j] * scale;
            network->Wf_m[a][j] = beta1 * network->Wf_m[a][j] + (1.0 - beta1) * g;
            network->Wf_v[a][j] = beta2 * network->Wf_v[a][j] + (1.0 - beta2) * (g * g);
            double mhat = network->Wf_m[a][j] * inv_bc1;
            double vhat = network->Wf_v[a][j] * inv_bc2;
            network->Wf[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
            
            g = dWi[a][j] * scale;
            network->Wi_m[a][j] = beta1 * network->Wi_m[a][j] + (1.0 - beta1) * g;
            network->Wi_v[a][j] = beta2 * network->Wi_v[a][j] + (1.0 - beta2) * (g * g);
            mhat = network->Wi_m[a][j] * inv_bc1;
            vhat = network->Wi_v[a][j] * inv_bc2;
            network->Wi[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
            
            g = dWc[a][j] * scale;
            network->Wc_m[a][j] = beta1 * network->Wc_m[a][j] + (1.0 - beta1) * g;
            network->Wc_v[a][j] = beta2 * network->Wc_v[a][j] + (1.0 - beta2) * (g * g);
            mhat = network->Wc_m[a][j] * inv_bc1;
            vhat = network->Wc_v[a][j] * inv_bc2;
            network->Wc[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
            
            g = dWo[a][j] * scale;
            network->Wo_m[a][j] = beta1 * network->Wo_m[a][j] + (1.0 - beta1) * g;
            network->Wo_v[a][j] = beta2 * network->Wo_v[a][j] + (1.0 - beta2) * (g * g);
            mhat = network->Wo_m[a][j] * inv_bc1;
            vhat = network->Wo_v[a][j] * inv_bc2;
            network->Wo[a][j] += learningRate * (mhat / (sqrt(vhat) + eps));
        }
    }
    for (int j = 0; j < H; j++) {
        for (int k = 0; k < O; k++) {
            double g = dWout[j][k] * scale;
            network->Wout_m[j][k] = beta1 * network->Wout_m[j][k] + (1.0 - beta1) * g;
            network->Wout_v[j][k] = beta2 * network->Wout_v[j][k] + (1.0 - beta2) * (g * g);
            double mhat = network->Wout_m[j][k] * inv_bc1;
            double vhat = network->Wout_v[j][k] * inv_bc2;
            network->Wout[j][k] += learningRate * (mhat / (sqrt(vhat) + eps));
        }
    }

    for (int j = 0; j < H; j++) {
        double g;
        g = dBf[j] * scale;
        network->Bf_m[j] = beta1 * network->Bf_m[j] + (1.0 - beta1) * g;
        network->Bf_v[j] = beta2 * network->Bf_v[j] + (1.0 - beta2) * (g * g);
        double mhat = network->Bf_m[j] * inv_bc1;
        double vhat = network->Bf_v[j] * inv_bc2;
        network->Bf[j] += learningRate * (mhat / (sqrt(vhat) + eps));

        g = dBi[j] * scale;
        network->Bi_m[j] = beta1 * network->Bi_m[j] + (1.0 - beta1) * g;
        network->Bi_v[j] = beta2 * network->Bi_v[j] + (1.0 - beta2) * (g * g);
        mhat = network->Bi_m[j] * inv_bc1;
        vhat = network->Bi_v[j] * inv_bc2;
        network->Bi[j] += learningRate * (mhat / (sqrt(vhat) + eps));

        g = dBc[j] * scale;
        network->Bc_m[j] = beta1 * network->Bc_m[j] + (1.0 - beta1) * g;
        network->Bc_v[j] = beta2 * network->Bc_v[j] + (1.0 - beta2) * (g * g);
        mhat = network->Bc_m[j] * inv_bc1;
        vhat = network->Bc_v[j] * inv_bc2;
        network->Bc[j] += learningRate * (mhat / (sqrt(vhat) + eps));

        g = dBo[j] * scale;
        network->Bo_m[j] = beta1 * network->Bo_m[j] + (1.0 - beta1) * g;
        network->Bo_v[j] = beta2 * network->Bo_v[j] + (1.0 - beta2) * (g * g);
        mhat = network->Bo_m[j] * inv_bc1;
        vhat = network->Bo_v[j] * inv_bc2;
        network->Bo[j] += learningRate * (mhat / (sqrt(vhat) + eps));
    }

    for (int k = 0; k < O; k++) {
        double g = dBout[k] * scale;
        network->Bout_m[k] = beta1 * network->Bout_m[k] + (1.0 - beta1) * g;
        network->Bout_v[k] = beta2 * network->Bout_v[k] + (1.0 - beta2) * (g * g);
        double mhat = network->Bout_m[k] * inv_bc1;
        double vhat = network->Bout_v[k] * inv_bc2;
        network->Bout[k] += learningRate * (mhat / (sqrt(vhat) + eps));
    }

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

    for (int b = 0; b < batchCount; b++) free(Gs[b]);
    free(Gs);
    free(allG);
}