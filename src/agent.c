#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "struct.h"
#include "func.h"
#include "state.h"
#include "reward.h"
#include "policy.h"

#include "../mGBA-interface/include/mgba_connection.h"
#include "../mGBA-interface/include/mgba_controller.h"
#include "../mGBA-interface/include/mgba_map.h"
#include "../mGBA-interface/include/mgba_intel.h"

#define ACTION_COUNT 8
static const MGBAButton ACTIONS[ACTION_COUNT] = {
    MGBA_BUTTON_UP, MGBA_BUTTON_DOWN, MGBA_BUTTON_LEFT, MGBA_BUTTON_RIGHT,
    MGBA_BUTTON_A, MGBA_BUTTON_B, MGBA_BUTTON_START, MGBA_BUTTON_SELECT
};

MGBAButton chooseAction(double* probs) {
    double r = (double)rand() / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 0; i < ACTION_COUNT; i++) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return ACTIONS[i];
        }
    }
    return ACTIONS[5];
}

trajectory* runTrajectory(MGBAConnection conn, LSTM* network, int steps) {
    trajectory* traj = initTrajectory(steps);
    assert(traj != NULL);

    for (int j=0; j<network->hiddenSize; j++) {
        network->hiddenState[j] = 0.0;
        network->cellState[j] = 0.0;
    }

    for (int i=0; i<steps; i++) {
        traj->states[i] = fetchState(conn);

        double* input_vec = convertState(traj->states[i]);
        double* hidden = forward(network, input_vec);
        double* distribution = softmaxLayer(hidden, ACTION_COUNT);

        traj->probs[i] = malloc(ACTION_COUNT * sizeof(double));
        assert(traj->probs[i] != NULL);

        for (int k=0; k<ACTION_COUNT; k++) traj->probs[i][k] = distribution[k]; 
        traj->actions[i] = chooseAction(traj->probs[i]);

        mgba_press_button(&conn, traj->actions[i], 50);

        state s_next = fetchState(conn);
        traj->rewards[i] = pnl(traj->states[i], s_next);

        free(distribution);
        free(input_vec);
        freeState(s_next);
    }

    return traj;
}

int main() {
    MGBAConnection conn;
    if (mgba_connect(&conn, "127.0.0.1", 8888) == 0) printf("Connected to mGBA.\n");

    srand((unsigned int)time(NULL));

    int inputSize = 4*(32*32) + 6*8 + 4 + 3 + 2;
    int hiddenSize = 8;
    LSTM* network = initLSTM(inputSize, hiddenSize);

    int trajectories = 30;
    int steps = 64;

    while (true) {
        mgba_reset(&conn);
        
        mgba_press_button(&conn, MGBA_BUTTON_B, 50);
        mgba_press_button(&conn, MGBA_BUTTON_B, 50);
        mgba_press_button(&conn, MGBA_BUTTON_B, 1000);

        mgba_press_button(&conn, MGBA_BUTTON_START, 1000);
        mgba_press_button(&conn, MGBA_BUTTON_START, 1000);

        mgba_press_button(&conn, MGBA_BUTTON_DOWN, 1000);
        mgba_press_button(&conn, MGBA_BUTTON_DOWN, 200);
        mgba_press_button(&conn, MGBA_BUTTON_DOWN, 200);
        mgba_press_button(&conn, MGBA_BUTTON_UP, 200);

        for (int i=0; i<256; i++) {
            mgba_press_button(&conn, MGBA_BUTTON_A, 50);
        }
        mgba_press_button(&conn, MGBA_BUTTON_B, 50);
        mgba_press_button(&conn, MGBA_BUTTON_B, 50);
        printf("Game Initialized\n\n");

        for (int t=0; t<trajectories; t++) {
            trajectory* traj = runTrajectory(conn, network, steps);
            
            double ret = 0.0;
            for (int i=0; i<steps; i++) ret += traj->rewards[i];
            printf("Trajectory %d: return=%.3f\n", t+1, ret);

            double* data = convertState(traj->states[0]);
            backpropagation(network, data, 0.01, steps, traj);
            
            free(data);
            freeTrajectory(traj);

            if (stop()) {
                printf("Objective met.\n");
                break;
            }
        }
    }

    freeLSTM(network);
    mgba_disconnect(&conn);
    return 0;
}