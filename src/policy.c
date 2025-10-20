#include "policy.h"

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

double** dL_dWf(LSTM* network, double* z, double* oArray, double* fArray) {
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_df_grad = dc_df(network->cellState, network->hiddenSize);
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
        case MGBA_BUTTON_SELECT: return 7;
        default: return 5;
    }
}

double** getAdvantageWf(LSTM* network, double* z, double* oArray, double* fArray, double** probs, MGBAButton* actions, int steps, double* reward) {
    double** output = malloc((network->hiddenSize + network->inputSize) * sizeof(double*));
    assert(output != NULL);
    for (int i = 0; i < (network->hiddenSize + network->inputSize); i++) {
        output[i] = calloc(network->hiddenSize, sizeof(double));
        assert(output[i] != NULL);
    }

    double** dlogits_dWf = dL_dWf(network, z, oArray, fArray);

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
    double* cellPrev = malloc(network->hiddenSize * sizeof(double));
    assert(cellPrev != NULL);
    for (int i = 0; i < network->hiddenSize; i++) {
        cellPrev[i] = network->cellState[i];
    }

    double* combinedState = malloc((network->hiddenSize + network->inputSize) * sizeof(double));
    assert(combinedState != NULL);

    for (int i = 0; i < network->inputSize; i++) {
        combinedState[i] = data[i];
    }
    for (int i = 0; i < network->hiddenSize; i++) combinedState[i + network->inputSize] = network->hiddenState[i];

    double* fArray = forgetGate(network, combinedState);
    double* iArray = inputGate(network, combinedState);
    double* gArray = cellGate(network, combinedState);
    double* oArray = outputGate(network, combinedState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * gArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    double* rewards = discountedPNL(trajectories->states, 0.9, steps);

    double** advantageWf = getAdvantageWf(network, combinedState, oArray, fArray, trajectories->probs, trajectories->actions, steps, rewards);
    double** advantageWi = getAdvantageWi(network, combinedState, oArray, gArray, iArray, trajectories->probs, trajectories->actions, steps, rewards);
    double** advantageWg = getAdvantageWg(network, combinedState, iArray, oArray, gArray, trajectories->probs, trajectories->actions, steps, rewards);
    double** advantageWo = getAdvantageWo(network, combinedState, oArray, trajectories->probs, trajectories->actions, steps, rewards);

    double* advantageBf = getAdvantageBf(network, cellPrev, fArray, oArray, trajectories->probs, trajectories->actions, steps, rewards);
    double* advantageBi = getAdvantageBi(network, iArray, gArray, oArray, trajectories->probs, trajectories->actions, steps, rewards);
    double* advantageBg = getAdvantageBg(network, iArray, gArray, oArray, trajectories->probs, trajectories->actions, steps, rewards);
    double* advantageBo = getAdvantageBo(network, oArray, trajectories->probs, trajectories->actions, steps, rewards);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        for (int j = 0; j < network->hiddenSize; j++) {
            network->Wf[i][j] += learningRate * advantageWf[i][j];
            network->Wi[i][j] += learningRate * advantageWi[i][j];
            network->Wc[i][j] += learningRate * advantageWg[i][j];
            network->Wo[i][j] += learningRate * advantageWo[i][j];
        }
    }

    for (int i = 0; i < network->hiddenSize; i++) {
        network->Bf[i] += learningRate * advantageBf[i];
        network->Bi[i] += learningRate * advantageBi[i];
        network->Bc[i] += learningRate * advantageBg[i];
        network->Bo[i] += learningRate * advantageBo[i];
    }

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(advantageWf[i]);
        free(advantageWi[i]);
        free(advantageWg[i]);
        free(advantageWo[i]);
    }
    free(advantageWf);
    free(advantageWi);
    free(advantageWg);
    free(advantageWo);

    free(advantageBf);
    free(advantageBi);
    free(advantageBg);
    free(advantageBo);

    free(cellPrev);
    free(combinedState);
    free(fArray);
    free(iArray);
    free(gArray);
    free(oArray);

    free(rewards);

    return network->hiddenState;
}