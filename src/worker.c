#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "struct.h"
#include "func.h"
#include "state.h"
#include "reward.h"
#include "policy.h"
#include "checkpoint.h"
#include "serializer.h"

#include "../mGBA-interface/include/mgba_connection.h"
#include "../mGBA-interface/include/mgba_controller.h"

static void ensure_dir(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static const MGBAButton ACTIONS[ACTION_COUNT] = {
    MGBA_BUTTON_UP, MGBA_BUTTON_DOWN, MGBA_BUTTON_LEFT, MGBA_BUTTON_RIGHT,
    MGBA_BUTTON_A, MGBA_BUTTON_B
};

static int chooseAction(double* p, int n) {
    double r = (double)rand() / RAND_MAX;
    double c = 0.0;
    for (int i = 0; i < n; i++) {
        c += p[i];
        if (r < c || i == n - 1) return i;
    }
    return n - 1;
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
        double* probs = forward(network, input_vec, temperature);

        traj->probs[i] = malloc(ACTION_COUNT * sizeof(double));
        assert(traj->probs[i] != NULL);
        for (int k=0; k<ACTION_COUNT; k++) traj->probs[i][k] = probs[k];
        
        double prob_eps[ACTION_COUNT];
        double uni = 1.0 / (double)ACTION_COUNT;
        for (int k=0; k<ACTION_COUNT; k++) prob_eps[k] = (1.0 - epsilon) * traj->probs[i][k] + epsilon * uni;

        traj->behav_probs[i] = malloc(ACTION_COUNT * sizeof(double));
        assert(traj->behav_probs[i] != NULL);
        for (int k=0; k<ACTION_COUNT; k++) traj->behav_probs[i][k] = prob_eps[k];

        traj->actions[i] =  ACTIONS[chooseAction(prob_eps, ACTION_COUNT)];

        mgba_press_button(&conn, traj->actions[i], 50);

        state s_next = fetchState(conn);
        traj->rewards[i] = pnl(traj->states[i], s_next);

        free(input_vec);
    }

    return traj;
}

int main(int argc, char** argv) {
    const char* host = "127.0.0.1";
    int port = 8888;
    const char* queue_dir = "queue";
    const char* checkpoint_path = "checkpoints/model-last.sav";

    if (argc >= 2) port = atoi(argv[1]);

    ensure_dir("checkpoints");
    ensure_dir(queue_dir);

    MGBAConnection conn;
    if (mgba_connect(&conn, host, port) == 0) {
        printf("[Worker] Connected to mGBA %s:%d\n", host, port);
    } else {
        printf("[Worker] Failed to connect mGBA %s:%d\n", host, port);
        return 1;
    }

    unsigned int seed = (unsigned int)time(NULL);
    srand(seed);

    int inputSize = 6*8 + 4 + 3 + 2 + 2*32*32;
    int hiddenSize = 128;

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(checkpoint_path, &loaded_episodes, &loaded_seed);
    if (!network) network = initLSTM(inputSize, hiddenSize, ACTION_COUNT);
    
    int trajectories = 32;
    int steps = 64;
    int batch_size = 8;

    int episode = (int)loaded_episodes;
    double temperature = 1.0;
    double epsilon = 0.2;

    int file_seq = 0;

    while (1) {
        uint64_t ep_tmp = 0ULL;
        uint64_t seed_tmp = 0ULL;
        LSTM* reloaded = loadLSTM(checkpoint_path, &ep_tmp, &seed_tmp);
        if (reloaded) {
            if (network) freeLSTM(network);
            network = reloaded;
            episode = (int)ep_tmp;
            if (seed_tmp != 0ULL) { 
                seed = (unsigned int)seed_tmp; 
                srand(seed); 
            }
            printf("[Worker] Reloaded checkpoint (episode=%d)\n", episode);
        }

        mgba_reset(&conn);

        temperature = fmax(1.0, 3.0 * pow(0.97, (double)episode));
        epsilon = fmax(0.02, 0.2 * pow(0.99, (double)episode));

        for (int t=0; t<trajectories; t += batch_size) {
            trajectory** batch = (trajectory**)malloc(batch_size * sizeof(trajectory*));
            assert(batch != NULL);

            for (int b=0; b<batch_size; b++) {
                batch[b] = runTrajectory(conn, network, steps, temperature, epsilon);
            }

            char tmp_path[512], final_path[512];
#ifdef _WIN32
            snprintf(tmp_path, sizeof(tmp_path), "%s\\worker-%d-%06d.traj.tmp", queue_dir, port, file_seq);
            snprintf(final_path, sizeof(final_path), "%s\\worker-%d-%06d.traj", queue_dir, port, file_seq);
#else
            snprintf(tmp_path, sizeof(tmp_path), "%s/worker-%d-%06d.traj.tmp", queue_dir, port, file_seq);
            snprintf(final_path, sizeof(final_path), "%s/worker-%d-%06d.traj", queue_dir, port, file_seq);
#endif

            if (write_batch_file(tmp_path, final_path, batch, batch_size, steps, epsilon, temperature) != 0) {
                fprintf(stderr, "[Worker] Failed to write %s\n", final_path);
            } else {
                printf("[Worker] Enqueued %s\n", final_path);
            }
            file_seq++;

            for (int b=0; b<batch_size; b++) freeTrajectory(batch[b]);
            free(batch);
        }

        episode++;
    }

    freeLSTM(network);
    mgba_disconnect(&conn);
    return 0;
}