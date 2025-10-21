#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "struct.h"
#include "func.h"
#include "state.h"
#include "reward.h"
#include "policy.h"
#include "checkpoint.h"

#include "../mGBA-interface/include/mgba_connection.h"
#include "../mGBA-interface/include/mgba_controller.h"
#include "../mGBA-interface/include/mgba_map.h"
#include "../mGBA-interface/include/mgba_intel.h"

static const MGBAButton ACTIONS[ACTION_COUNT] = {
    MGBA_BUTTON_UP, MGBA_BUTTON_DOWN, MGBA_BUTTON_LEFT, MGBA_BUTTON_RIGHT,
    MGBA_BUTTON_A, MGBA_BUTTON_B, MGBA_BUTTON_START
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

trajectory* runTrajectory(MGBAConnection conn, LSTM* network, int steps, double temperature, double epsilon) {
    trajectory* traj = initTrajectory(steps);
    assert(traj != NULL);

    for (int j=0; j<network->hiddenSize; j++) {
        network->hiddenState[j] = 0.0;
        network->cellState[j] = 0.0;
    }

    for (int i=0; i<steps; i++) {
        traj->states[i] = fetchState(conn);

        double* input_vec = convertState(traj->states[i]);
        double* distribution = forward(network, input_vec, temperature);

        traj->probs[i] = malloc(ACTION_COUNT * sizeof(double));
        assert(traj->probs[i] != NULL);

        for (int k=0; k<ACTION_COUNT; k++) traj->probs[i][k] = distribution[k];
        if (((double)rand() / RAND_MAX) < epsilon) {
            traj->actions[i] = ACTIONS[rand() % ACTION_COUNT];
        } else {
            traj->actions[i] = chooseAction(distribution);
        }

        mgba_press_button(&conn, traj->actions[i], 50);

        state s_next = fetchState(conn);
        traj->rewards[i] = pnl(traj->states[i], s_next);

        freeState(s_next);
    }

    return traj;
}

int main() {
    MGBAConnection conn;
    if (mgba_connect(&conn, "127.0.0.1", 8888) == 0) printf("Connected to mGBA.\n");

    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);

    int inputSize = 4*(32*32) + 6*8 + 4 + 3 + 2;
    int hiddenSize = 128;

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM("checkpoints/model-last.bin", &loaded_episodes, &loaded_seed);
    if (network) {
        printf("Loaded model from checkpoints/model-last.bin (input=%d, hidden=%d)\n", network->inputSize, network->hiddenSize);
    } else {
    network = initLSTM(inputSize, hiddenSize, ACTION_COUNT);
        printf("Initialized new model (input=%d, hidden=%d)\n", inputSize, hiddenSize);
    }

    int trajectories = 32;
    int steps = 64;
    int batch_size = 8;

    int episode = 0;

    double temperature = 1.0;
    double epsilon = 0.2;

    if (network && loaded_episodes > 0ULL) {
        episode = (int)loaded_episodes;
        if (loaded_seed != 0ULL) {
            seed = (unsigned int)loaded_seed;
            srand(seed);
        }
    }

    while (true) {
        mgba_reset(&conn);
        
        mgba_press_button(&conn, MGBA_BUTTON_B, 3000);
        mgba_press_button(&conn, MGBA_BUTTON_B, 500);
        mgba_press_button(&conn, MGBA_BUTTON_B, 2000);

        mgba_press_button(&conn, MGBA_BUTTON_START, 1000);
        mgba_press_button(&conn, MGBA_BUTTON_START, 2000);

        mgba_press_button(&conn, MGBA_BUTTON_DOWN, 500);
        mgba_press_button(&conn, MGBA_BUTTON_DOWN, 500);
        mgba_press_button(&conn, MGBA_BUTTON_DOWN, 500);
        mgba_press_button(&conn, MGBA_BUTTON_UP, 500);

        for (int i=0; i<300; i++) {
            mgba_press_button(&conn, MGBA_BUTTON_A, 50);
        }
        mgba_press_button(&conn, MGBA_BUTTON_B, 50);
        mgba_press_button(&conn, MGBA_BUTTON_B, 50);
        printf("Game Initialized\n\n");

        temperature = fmax(1.0, 3.0 * pow(0.97, (double)episode));
        epsilon = fmax(0.02, 0.2 * pow(0.99, (double)episode));
        
        for (int t=0; t<trajectories; t += batch_size) {
            int m = (t + batch_size <= trajectories) ? batch_size : (trajectories - t);
            trajectory** batch = malloc(m * sizeof(trajectory*));
            assert(batch != NULL);

            for (int b=0; b<m; b++) {
                int idx = t + b;
                batch[b] = runTrajectory(conn, network, steps, temperature, epsilon);
                double ret = 0.0;
                for (int i=0; i<steps; i++) ret += batch[b]->rewards[i];
                printf("Trajectory %d: return=%lf\n", idx+1, ret);
            }

            backpropagation(network, NULL, 0.01, steps, batch, m, temperature, epsilon);

            for (int b=0; b<m; b++) freeTrajectory(batch[b]);
            free(batch);

            if (stop()) {
                printf("Objective met.\n");
                break;
            }
        }

        episode++;
        printf("\nEpisode %d completed.\n", episode);

        #ifdef _WIN32
        _mkdir("checkpoints");
        #else
        mkdir("checkpoints", 0755);
        #endif
        if (saveLSTMCheckpoint("checkpoints/model-last.bin", network, (uint64_t)episode, (uint64_t)seed) == 0) {
            printf("Saved checkpoint: checkpoints/model-last.bin (episode=%d)\n", episode);
        } else {
            printf("Warning: failed to save checkpoint.\n");
        }

        reset_flags();
    }

    freeLSTM(network);
    mgba_disconnect(&conn);
    return 0;
}