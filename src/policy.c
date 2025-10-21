#include "policy.h"
#include "state.h"

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

double* forward(LSTM* network, double* data) {
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

    free(combinedState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);

    return network->hiddenState;
}

double* dh_dc(double* oArray, double* cellState, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = oArray[i] * (1 - tanh(cellState[i]) * tanh(cellState[i]));
    }

    return grad;
}

double* dh_do(double* cellState, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = tanh(cellState[i]);
    }

    return grad;
}

double* dc_df(double* cellPrev, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = cellPrev[i];
    }

    return grad;
}

double* dc_di(double* gArray, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = gArray[i];
    }

    return grad;
}

double* dc_dg(double* iArray, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = iArray[i];
    }

    return grad;
}

double** df_dWf(LSTM* network, double* fArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = fArray[j] * (1 - fArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}

double** di_dWi(LSTM* network, double* iArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = iArray[j] * (1 - iArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}

double** dg_dWg(LSTM* network, double* gArray, double* z) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = (1 - gArray[j] * gArray[j]) * z[i];
        }
    }

    return grad;
}

double** do_dWo(LSTM* network, double* oArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = oArray[j] * (1 - oArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}

double** dL_dWf(LSTM* network, double* z, double* oArray, double* fArray, double* cellPrev) {
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_df_grad = dc_df(cellPrev, network->hiddenSize);
    double** df_dWf_grad = df_dWf(network, fArray, z);

    double** dL_dWf = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWf != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWf[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWf[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWf[i][j] = dh_dc_grad[j] * dc_df_grad[j] * df_dWf_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(df_dWf_grad[i]);
    }
    free(dh_dc_grad);
    free(dc_df_grad);
    free(df_dWf_grad);

    return dL_dWf;
}

double** dL_dWi(LSTM* network, double* z, double* oArray,double* gArray, double* iArray) {
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_di_grad = dc_di(gArray, network->hiddenSize);
    double** di_dWi_grad = di_dWi(network, iArray, z);

    double** dL_dWi = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWi != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWi[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWi[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWi[i][j] = dh_dc_grad[j] * dc_di_grad[j] * di_dWi_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(di_dWi_grad[i]);
    }
    free(dh_dc_grad);
    free(dc_di_grad);
    free(di_dWi_grad);

    return dL_dWi;
}

double** dL_dWg(LSTM* network, double* z, double* iArray, double* oArray, double* gArray) {
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_dg_grad = dc_dg(iArray, network->hiddenSize);
    double** dg_dWg_grad = dg_dWg(network, gArray, z);

    double** dL_dWg = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWg != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWg[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWg[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWg[i][j] = dh_dc_grad[j] * dc_dg_grad[j] * dg_dWg_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(dg_dWg_grad[i]);
    }
    free(dh_dc_grad);
    free(dc_dg_grad);
    free(dg_dWg_grad);

    return dL_dWg;
}

double** dL_dWo(LSTM* network, double* z, double* oArray) {
    double* dc_do_grad = dh_do(network->cellState, network->hiddenSize);
    double** do_dWo_grad = do_dWo(network, oArray, z);

    double** dL_dWo = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWo != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWo[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWo[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWo[i][j] = dc_do_grad[j] * do_dWo_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(do_dWo_grad[i]);
    }
    free(dc_do_grad);
    free(do_dWo_grad);

    return dL_dWo;
}

double* dc_dBf(double* cellPrev, double* fArray, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = cellPrev[i] * fArray[i] * (1 - fArray[i]);
    }

    return grad;
}

double* dc_dBi(double* gArray, double* iArray, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = gArray[i] * iArray[i] * (1 - iArray[i]);
    }

    return grad;
}

double* dc_dBg(double* iArray, double* gArray, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = iArray[i] * (1 - gArray[i] * gArray[i]);
    }

    return grad;
}

double* dh_dBo(double* oArray, double* cellState, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = oArray[i] * (1 - oArray[i]) * tanh(cellState[i]);
    }

    return grad;
}

double* dL_dBf(double* cellPrev, double* cellState, double* fArray, double* oArray, int hiddenSize) {
    double* dh_dc_grad = dh_dc(oArray, cellState, hiddenSize);
    double* dc_dBf_grad = dc_dBf(cellPrev, fArray, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dh_dc_grad[i] * dc_dBf_grad[i];
    }

    free(dh_dc_grad);
    free(dc_dBf_grad);

    return grad;
}

double* dL_dBi(double* cellState, double* iArray, double* gArray, double* oArray, int hiddenSize) {
    double* dh_dc_grad = dh_dc(oArray, cellState, hiddenSize);
    double* dc_dBi_grad = dc_dBi(gArray, iArray, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dh_dc_grad[i] * dc_dBi_grad[i];
    }

    free(dh_dc_grad);
    free(dc_dBi_grad);

    return grad;
}

double* dL_dBg(double* cellState, double* iArray, double* gArray, double* oArray, int hiddenSize) {
    double* dh_dc_grad = dh_dc(oArray, cellState, hiddenSize);
    double* dc_dBg_grad = dc_dBg(iArray, gArray, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dh_dc_grad[i] * dc_dBg_grad[i];
    }

    free(dh_dc_grad);
    free(dc_dBg_grad);

    return grad;
}

double* dL_dBo(double* cellState, double* oArray, int hiddenSize) {
    double* dh_dBo_grad = dh_dBo(oArray, cellState, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dh_dBo_grad[i];
    }

    free(dh_dBo_grad);

    return grad;
}

double* discountedPNL(state* etats, double gamma, int steps) {
    double* G = calloc(steps, sizeof(double));
    assert(G != NULL);
    
    for (int t = steps - 2; t >= 0; t--) {
        double r = pnl(etats[t], etats[t+1]);
        G[t] = r + gamma * G[t+1];
    }

    return G;
}

double* softmaxLayer(double* logits, int n) {
    double* probs = malloc(n * sizeof(double));
    assert(probs != NULL);

    double sum = 0.0;
    for (int i=0; i<n; i++) {
        probs[i] = exp(logits[i]);
        sum += probs[i];
    }
    for (int i=0; i<n; i++) {
        probs[i] /= sum;
    }

    return probs;
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

double** getAdvantageWf(LSTM* network, double* z, double* oArray, double* fArray, double* cellPrev, double** probs, MGBAButton* actions, int steps, double* reward) {
    double** output = malloc((network->hiddenSize + network->inputSize) * sizeof(double*));
    assert(output != NULL);
    for (int i = 0; i < (network->hiddenSize + network->inputSize); i++) {
        output[i] = calloc(network->hiddenSize, sizeof(double));
        assert(output[i] != NULL);
    }

    double** dlogits_dWf = dL_dWf(network, z, oArray, fArray, cellPrev);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);

        for (int i=0; i<(network->hiddenSize + network->inputSize); i++) {
            for (int j=0; j<network->hiddenSize; j++) {
                double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
                output[i][j] += dlogits_dWf[i][j] * r * grad_logit_j;
            }
        }
    }

    for (int i = 0; i < (network->hiddenSize + network->inputSize); i++) {
        free(dlogits_dWf[i]);
    }
    free(dlogits_dWf);

    return output;
}

double** getAdvantageWi(LSTM* network, double* z, double* oArray, double* gArray, double* iArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double** output = malloc((network->hiddenSize + network->inputSize) * sizeof(double*));
    assert(output != NULL);
    for (int i = 0; i < (network->hiddenSize + network->inputSize); i++) {
        output[i] = calloc(network->hiddenSize, sizeof(double));
        assert(output[i] != NULL);
    }

    double** dlogits_dWi = dL_dWi(network, z, oArray, gArray, iArray);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int i=0; i<(network->hiddenSize + network->inputSize); i++) {
            for (int j=0; j<network->hiddenSize; j++) {
                double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
                output[i][j] += dlogits_dWi[i][j] * (r * grad_logit_j);
            }
        }
    }

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(dlogits_dWi[i]);
    }
    free(dlogits_dWi);

    return output;
}

double** getAdvantageWg(LSTM* network, double* z, double* iArray, double* oArray, double* gArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double** output = malloc((network->hiddenSize + network->inputSize) * sizeof(double*));
    assert(output != NULL);
    for (int i = 0; i < (network->hiddenSize + network->inputSize); i++) {
        output[i] = calloc(network->hiddenSize, sizeof(double));
        assert(output[i] != NULL);
    }

    double** dlogits_dWg = dL_dWg(network, z, iArray, oArray, gArray);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int i=0; i<(network->hiddenSize + network->inputSize); i++) {
            for (int j=0; j<network->hiddenSize; j++) {
                double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
                output[i][j] += dlogits_dWg[i][j] * (r * grad_logit_j);
            }
        }
    }

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(dlogits_dWg[i]);
    }
    free(dlogits_dWg);

    return output;
}

double** getAdvantageWo(LSTM* network, double* z, double* oArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double** output = malloc((network->hiddenSize + network->inputSize) * sizeof(double*));
    assert(output != NULL);
    for (int i = 0; i < (network->hiddenSize + network->inputSize); i++) {
        output[i] = calloc(network->hiddenSize, sizeof(double));
        assert(output[i] != NULL);
    }

    double** dlogits_dWo = dL_dWo(network, z, oArray);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int i=0; i<(network->hiddenSize + network->inputSize); i++) {
            for (int j=0; j<network->hiddenSize; j++) {
                double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
                output[i][j] += dlogits_dWo[i][j] * (r * grad_logit_j);
            }
        }
    }

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(dlogits_dWo[i]);
    }
    free(dlogits_dWo);

    return output;
}

double* getAdvantageBf(LSTM* network, double* cellPrev, double* fArray, double* oArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double* output = calloc(network->hiddenSize, sizeof(double));
    assert(output != NULL);

    double* dlogits_dBf = dL_dBf(cellPrev, network->cellState, fArray, oArray, network->hiddenSize);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int j=0; j<network->hiddenSize; j++) {
            double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
            output[j] += dlogits_dBf[j] * (r * grad_logit_j);
        }
    }

    free(dlogits_dBf);

    return output;
}

double* getAdvantageBi(LSTM* network, double* iArray, double* gArray, double* oArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double* output = calloc(network->hiddenSize, sizeof(double));
    assert(output != NULL);

    double* dlogits_dBi = dL_dBi(network->cellState, iArray, gArray, oArray, network->hiddenSize);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int j=0; j<network->hiddenSize; j++) {
            double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
            output[j] += dlogits_dBi[j] * (r * grad_logit_j);
        }
    }

    free(dlogits_dBi);

    return output;
}

double* getAdvantageBo(LSTM* network, double* oArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double* output = calloc(network->hiddenSize, sizeof(double));
    assert(output != NULL);

    double* dlogits_dBo = dL_dBo(network->cellState, oArray, network->hiddenSize);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int j=0; j<network->hiddenSize; j++) {
            double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
            output[j] += dlogits_dBo[j] * (r * grad_logit_j);
        }
    }

    free(dlogits_dBo);

    return output;
}

double* getAdvantageBg(LSTM* network, double* iArray, double* gArray, double* oArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double* output = calloc(network->hiddenSize, sizeof(double));
    assert(output != NULL);

    double* dlogits_dBg = dL_dBg(network->cellState, iArray, gArray, oArray, network->hiddenSize);

    for (int k=0; k<steps; k++) {
        double r = reward[k];
        int actionIndex = actionToIndex(actions[k]);
        for (int j=0; j<network->hiddenSize; j++) {
            double grad_logit_j = ((j == actionIndex) ? 1.0 : 0.0) - probs[k][j];
            output[j] += dlogits_dBg[j] * (r * grad_logit_j);
        }
    }

    free(dlogits_dBg);

    return output;
}

double* backpropagation(LSTM* network, double* data, double learningRate, int steps, trajectory* trajectories) {
    (void)data;
    int H = network->hiddenSize;
    int I = network->inputSize;
    int Z = I + H;
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

    double* G = discountedPNL(trajectories->states, 0.9, T);

    double** dWf = malloc(Z * sizeof(double*));
    double** dWi = malloc(Z * sizeof(double*));
    double** dWc = malloc(Z * sizeof(double*));
    double** dWo = malloc(Z * sizeof(double*));
    assert(dWf != NULL && dWi != NULL && dWc != NULL && dWo != NULL);

    for (int a = 0; a < Z; a++) {
        dWf[a] = calloc(H, sizeof(double));
        dWi[a] = calloc(H, sizeof(double));
        dWc[a] = calloc(H, sizeof(double));
        dWo[a] = calloc(H, sizeof(double));
        assert(dWf[a] != NULL && dWi[a] != NULL && dWc[a] != NULL && dWo[a] != NULL);
    }

    double* dBf = calloc(H, sizeof(double));
    double* dBi = calloc(H, sizeof(double));
    double* dBc = calloc(H, sizeof(double));
    double* dBo = calloc(H, sizeof(double));
    assert(dBf != NULL && dBi != NULL && dBc != NULL && dBo != NULL);

    double* dh_next = calloc(H, sizeof(double));
    double* dc_next = calloc(H, sizeof(double));
    assert(dh_next != NULL && dc_next != NULL);

    for (int t = T - 1; t >= 0; t--) {
        int actionIndex = actionToIndex(trajectories->actions[t]);
        double* dlogits = calloc(H, sizeof(double));
        assert(dlogits != NULL);
        for (int j = 0; j < H; j++) {
            double gadv = ((j == actionIndex) ? 1.0 : 0.0) - trajectories->probs[t][j];
            dlogits[j] = gadv * G[t];
        }

        double* dh = malloc(H * sizeof(double));
        assert(dh != NULL);
        for (int j = 0; j < H; j++) dh[j] = dlogits[j] + dh_next[j];

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

    for (int a = 0; a < Z; a++) {
        for (int j = 0; j < H; j++) network->Wf[a][j] += learningRate * dWf[a][j];
        for (int j = 0; j < H; j++) network->Wi[a][j] += learningRate * dWi[a][j];
        for (int j = 0; j < H; j++) network->Wc[a][j] += learningRate * dWc[a][j];
        for (int j = 0; j < H; j++) network->Wo[a][j] += learningRate * dWo[a][j];
    }

    for (int j = 0; j < H; j++) network->Bf[j] += learningRate * dBf[j];
    for (int j = 0; j < H; j++) network->Bi[j] += learningRate * dBi[j];
    for (int j = 0; j < H; j++) network->Bc[j] += learningRate * dBc[j];
    for (int j = 0; j < H; j++) network->Bo[j] += learningRate * dBo[j];

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

    free(dWf);
    free(dWi);
    free(dWc);
    free(dWo);
    free(dBf);
    free(dBi);
    free(dBc);
    free(dBo);
    free(hprev);
    free(cprev_vec);
    free(dh_next);
    free(dc_next);
    free(G);

    return network->hiddenState;
}