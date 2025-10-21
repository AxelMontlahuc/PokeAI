#include "policy.h"
#include "state.h"

static const double ENTROPY_COEFF = 0.01;

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

    double sum = 0.0;
    for (int k=0; k<network->outputSize; k++) {
        network->probs[k] = exp(network->logits[k] / temperature);
        sum += network->probs[k];
    }
    for (int k=0; k<network->outputSize; k++) {
        network->probs[k] /= sum;
    }

    free(combinedState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);

    return network->probs;
}


double* discountedPNL(double* rewards, double gamma, int steps, bool normalize) {
    double* G = calloc(steps, sizeof(double));
    assert(G != NULL);
    
    for (int t = steps - 2; t >= 0; t--) {
        G[t] = rewards[t] + gamma * G[t+1];
    }

    if (normalize) {
        double mean = 0.0;
        for (int t = 0; t < steps; t++) mean += G[t];
        mean /= (double)steps;
        double var = 0.0;
        for (int t = 0; t < steps; t++) {
            double d = G[t] - mean;
            var += d * d;
        }
        var /= (double)steps;
        double std = sqrt(var) + 1e-8;
        for (int t = 0; t < steps; t++) G[t] = (G[t] - mean) / std;
    }

    return G;
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


double* backpropagation(LSTM* network, double* data, double learningRate, int steps, trajectory* trajectories, double temperature, double epsilon) {
    (void)data;
    int H = network->hiddenSize;
    int I = network->inputSize;
    int Z = I + H;
    int O = network->outputSize;
    int T = steps;

    double** x = malloc(T * sizeof(double*));
    double** z = malloc(T * sizeof(double*));
    double** f = malloc(T * sizeof(double*));
    double** i = malloc(T * sizeof(double*));
    double** g = malloc(T * sizeof(double*));
    double** o = malloc(T * sizeof(double*));
    double** c = malloc(T * sizeof(double*));
    double** h = malloc(T * sizeof(double*));
    double** cprev = malloc(T * sizeof(double*));
    assert(x != NULL && z != NULL && f != NULL && i != NULL && g != NULL && o != NULL && c != NULL && h != NULL && cprev != NULL);

    double* hprev = calloc(H, sizeof(double));
    double* cprev_vec = calloc(H, sizeof(double));
    assert(hprev != NULL && cprev_vec != NULL);

    for (int t = 0; t < T; t++) {
        x[t] = convertState(trajectories->states[t]);
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
            double s_f = network->Bf[j];
            for (int a = 0; a < Z; a++) s_f += z[t][a] * network->Wf[a][j];
            f[t][j] = sigmoid(s_f);
        }
        for (int j = 0; j < H; j++) {
            double s_i = network->Bi[j];
            for (int a = 0; a < Z; a++) s_i += z[t][a] * network->Wi[a][j];
            i[t][j] = sigmoid(s_i);
        }
        for (int j = 0; j < H; j++) {
            double s_g = network->Bc[j];
            for (int a = 0; a < Z; a++) s_g += z[t][a] * network->Wc[a][j];
            g[t][j] = tanh(s_g);
        }
        for (int j = 0; j < H; j++) {
            double s_o = network->Bo[j];
            for (int a = 0; a < Z; a++) s_o += z[t][a] * network->Wo[a][j];
            o[t][j] = sigmoid(s_o);
        }
        for (int j = 0; j < H; j++) c[t][j] = f[t][j] * cprev_vec[j] + i[t][j] * g[t][j];
        for (int j = 0; j < H; j++) h[t][j] = o[t][j] * tanh(c[t][j]);

        for (int j = 0; j < H; j++) hprev[j] = h[t][j];
        for (int j = 0; j < H; j++) cprev_vec[j] = c[t][j];
    }

    double* G = discountedPNL(trajectories->rewards, 0.9, T, true);

    double** dWf = malloc(Z * sizeof(double*));
    double** dWi = malloc(Z * sizeof(double*));
    double** dWc = malloc(Z * sizeof(double*));
    double** dWo = malloc(Z * sizeof(double*));
    double** dWout = malloc(H * sizeof(double*));
    assert(dWf != NULL && dWi != NULL && dWc != NULL && dWo != NULL);
    assert(dWout != NULL);

    for (int a = 0; a < Z; a++) {
        dWf[a] = calloc(H, sizeof(double));
        dWi[a] = calloc(H, sizeof(double));
        dWc[a] = calloc(H, sizeof(double));
        dWo[a] = calloc(H, sizeof(double));
        assert(dWf[a] != NULL && dWi[a] != NULL && dWc[a] != NULL && dWo[a] != NULL);
    }

    for (int j = 0; j < H; j++) {
        dWout[j] = calloc(O, sizeof(double));
        assert(dWout[j] != NULL);
    }

    double* dBf = calloc(H, sizeof(double));
    double* dBi = calloc(H, sizeof(double));
    double* dBc = calloc(H, sizeof(double));
    double* dBo = calloc(H, sizeof(double));
    double* dBout = calloc(O, sizeof(double));
    assert(dBf != NULL && dBi != NULL && dBc != NULL && dBo != NULL && dBout != NULL);

    double* dh_next = calloc(H, sizeof(double));
    double* dc_next = calloc(H, sizeof(double));
    assert(dh_next != NULL && dc_next != NULL);

    for (int t = T - 1; t >= 0; t--) {
        int actionIndex = actionToIndex(trajectories->actions[t]);
        double* dlogits = calloc(O, sizeof(double));
        assert(dlogits != NULL);
        for (int k = 0; k < O; k++) {
            double gadv = ((k == actionIndex) ? 1.0 : 0.0) - trajectories->probs[t][k];
            dlogits[k] = gadv * G[t];
        }

        if (ENTROPY_COEFF > 0.0) {
            double mean_logp = 0.0;
            for (int k = 0; k < O; k++) {
                double pk = trajectories->probs[t][k];
                mean_logp += pk * log(pk + 1e-12);
            }
            for (int k = 0; k < O; k++) {
                double pk = trajectories->probs[t][k];
                double ent_grad = pk * (mean_logp - log(pk + 1e-12));
                dlogits[k] += ENTROPY_COEFF * ent_grad;
            }
        }

        double inv_temp = 1.0 / temperature;
        double scale_eps = fmax(0.0, 1.0 - epsilon);
        for (int k = 0; k < O; k++) dlogits[k] *= (inv_temp * scale_eps);

        double* dh = malloc(H * sizeof(double));
        assert(dh != NULL);
        for (int j = 0; j < H; j++) dh[j] = dh_next[j];
        dL_dWout(network, dlogits, h[t], dWout, dBout, dh);

        double* do_vec = malloc(H * sizeof(double));
        assert(do_vec != NULL);
        for (int j = 0; j < H; j++) do_vec[j] = dh[j] * tanh(c[t][j]);

        double* d_o_pre = malloc(H * sizeof(double));
        assert(d_o_pre != NULL);
        for (int j = 0; j < H; j++) d_o_pre[j] = do_vec[j] * o[t][j] * (1.0 - o[t][j]);

        double* dc = malloc(H * sizeof(double));
        assert(dc != NULL);
        for (int j = 0; j < H; j++) dc[j] = dh[j] * o[t][j] * (1.0 - tanh(c[t][j]) * tanh(c[t][j])) + dc_next[j];

        double* df = malloc(H * sizeof(double));
        assert(df != NULL);
        for (int j = 0; j < H; j++) df[j] = dc[j] * cprev[t][j];

        double* di_vec = malloc(H * sizeof(double));
        assert(di_vec != NULL);
        for (int j = 0; j < H; j++) di_vec[j] = dc[j] * g[t][j];

        double* dg = malloc(H * sizeof(double));
        assert(dg != NULL);
        for (int j = 0; j < H; j++) dg[j] = dc[j] * i[t][j];

        double* d_f_pre = malloc(H * sizeof(double));
        assert(d_f_pre != NULL);
        for (int j = 0; j < H; j++) d_f_pre[j] = df[j] * f[t][j] * (1.0 - f[t][j]);

        double* d_i_pre = malloc(H * sizeof(double));
        assert(d_i_pre != NULL);
        for (int j = 0; j < H; j++) d_i_pre[j] = di_vec[j] * i[t][j] * (1.0 - i[t][j]);

        double* d_g_pre = malloc(H * sizeof(double));
        assert(d_g_pre != NULL);
        for (int j = 0; j < H; j++) d_g_pre[j] = dg[j] * (1.0 - g[t][j] * g[t][j]);

        for (int j = 0; j < H; j++) dBf[j] += d_f_pre[j];
        for (int j = 0; j < H; j++) dBi[j] += d_i_pre[j];
        for (int j = 0; j < H; j++) dBc[j] += d_g_pre[j];
        for (int j = 0; j < H; j++) dBo[j] += d_o_pre[j];

        for (int a = 0; a < Z; a++) {
            for (int j = 0; j < H; j++) dWf[a][j] += z[t][a] * d_f_pre[j];
            for (int j = 0; j < H; j++) dWi[a][j] += z[t][a] * d_i_pre[j];
            for (int j = 0; j < H; j++) dWc[a][j] += z[t][a] * d_g_pre[j];
            for (int j = 0; j < H; j++) dWo[a][j] += z[t][a] * d_o_pre[j];
        }

        double* dz = calloc(Z, sizeof(double));
        assert(dz != NULL);
        for (int a = 0; a < Z; a++) {
            double sum = 0.0;
            for (int j = 0; j < H; j++) sum += network->Wf[a][j] * d_f_pre[j];
            dz[a] += sum;
        }
        for (int a = 0; a < Z; a++) {
            double sum = 0.0;
            for (int j = 0; j < H; j++) sum += network->Wi[a][j] * d_i_pre[j];
            dz[a] += sum;
        }
        for (int a = 0; a < Z; a++) {
            double sum = 0.0;
            for (int j = 0; j < H; j++) sum += network->Wc[a][j] * d_g_pre[j];
            dz[a] += sum;
        }
        for (int a = 0; a < Z; a++) {
            double sum = 0.0;
            for (int j = 0; j < H; j++) sum += network->Wo[a][j] * d_o_pre[j];
            dz[a] += sum;
        }

        for (int j = 0; j < H; j++) dh_next[j] = dz[I + j];
        for (int j = 0; j < H; j++) dc_next[j] = dc[j] * f[t][j];

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
    }

    double clip = 1.0;
    double norm2 = 0.0;
    for (int a = 0; a < Z; a++) {
        for (int j = 0; j < H; j++) norm2 += dWf[a][j]*dWf[a][j];
        for (int j = 0; j < H; j++) norm2 += dWi[a][j]*dWi[a][j];
        for (int j = 0; j < H; j++) norm2 += dWc[a][j]*dWc[a][j];
        for (int j = 0; j < H; j++) norm2 += dWo[a][j]*dWo[a][j];
    }
    for (int j = 0; j < H; j++) {
        for (int k = 0; k < O; k++) norm2 += dWout[j][k]*dWout[j][k];
    }
    for (int j = 0; j < H; j++) norm2 += dBf[j]*dBf[j];
    for (int j = 0; j < H; j++) norm2 += dBi[j]*dBi[j];
    for (int j = 0; j < H; j++) norm2 += dBc[j]*dBc[j];
    for (int j = 0; j < H; j++) norm2 += dBo[j]*dBo[j];
    for (int k = 0; k < O; k++) norm2 += dBout[k]*dBout[k];
    double norm = sqrt(norm2);
    double scale = (norm > clip) ? (clip / (norm + 1e-12)) : 1.0;

    for (int a = 0; a < Z; a++) {
        for (int j = 0; j < H; j++) network->Wf[a][j] += learningRate * (dWf[a][j] * scale);
        for (int j = 0; j < H; j++) network->Wi[a][j] += learningRate * (dWi[a][j] * scale);
        for (int j = 0; j < H; j++) network->Wc[a][j] += learningRate * (dWc[a][j] * scale);
        for (int j = 0; j < H; j++) network->Wo[a][j] += learningRate * (dWo[a][j] * scale);
    }

    for (int j = 0; j < H; j++) {
        for (int k = 0; k < O; k++) network->Wout[j][k] += learningRate * (dWout[j][k] * scale);
    }

    for (int j = 0; j < H; j++) network->Bf[j] += learningRate * (dBf[j] * scale);
    for (int j = 0; j < H; j++) network->Bi[j] += learningRate * (dBi[j] * scale);
    for (int j = 0; j < H; j++) network->Bc[j] += learningRate * (dBc[j] * scale);
    for (int j = 0; j < H; j++) network->Bo[j] += learningRate * (dBo[j] * scale);
    for (int k = 0; k < O; k++) network->Bout[k] += learningRate * (dBout[k] * scale);

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
    free(hprev);
    free(cprev_vec);
    free(dh_next);
    free(dc_next);
    free(G);

    return network->hiddenState;
}