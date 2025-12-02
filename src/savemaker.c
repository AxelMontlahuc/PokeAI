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
#include <sys/time.h>
#endif

#include "state.h"
#include "reward.h"
#include "policy.h"
#include "constants.h"
#include "checkpoint.h"
#include "serializer.h"
#include "../gba/gba.h"

static void ensure_dir(const char* path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static const int ACTIONS[ACTION_COUNT] = {
    0, 1, 2, 3, 4, 5 // UP, DOWN, LEFT, RIGHT, A, B
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

void startProcedure() {
    gba_run(300);
    for (int i=0; i<100; i++) {
        gba_button(4);
        gba_run(SPEED);
    }
    gba_screen(SCREEN_PATH);
}

trajectory* runTrajectory(LSTM* network, int steps, double temperature) {
    trajectory* traj = initTrajectory(steps);
    assert(traj != NULL);

    double* input_vec = malloc(INPUT_SIZE * sizeof(double));
    assert(input_vec != NULL);

    for (int i=0; i<steps; i++) {
        traj->states[i] = fetchState();
        convertState(traj->states[i], input_vec);
        double* probs = forward(network, input_vec, temperature);

        traj->probs[i] = malloc(ACTION_COUNT * sizeof(double));
        assert(traj->probs[i] != NULL);
        for (int k=0; k<ACTION_COUNT; k++) traj->probs[i][k] = probs[k];
        
        traj->actions[i] =  ACTIONS[chooseAction(traj->probs[i], ACTION_COUNT)];
        traj->values[i] = network->last_value;

        gba_button(traj->actions[i]);
        gba_run(SPEED);
        gba_screen(SCREEN_PATH);

        state s_next = fetchState();
        traj->rewards[i] = pnl(traj->states[i], s_next);
    }

    free(input_vec);

    return traj;
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        ID = atoi(argv[1]);
    }
    snprintf(SCREEN_PATH, sizeof(SCREEN_PATH), "%s%d.bmp", SCREEN_PATH_PREFIX, ID);

    ensure_dir(CHECKPOINT_DIR);
    ensure_dir(QUEUE_DIR);
    ensure_dir(LOCKS_DIR);
    ensure_dir(SAVES_DIR);
    ensure_dir(LOGS_DIR);
    ensure_dir(SCREEN_DIR);

    unsigned int seed = (unsigned int)(time(NULL) * ID);
    srand(seed);

    uint64_t loaded_episodes = 0ULL;
    uint64_t loaded_seed = 0ULL;

    LSTM* network = loadLSTM(CHECKPOINT_PATH, &loaded_episodes, &loaded_seed);
    if (!network) network = initLSTM(INPUT_SIZE, HIDDEN_SIZE, ACTION_COUNT);

    int episode = (int)loaded_episodes;
    double temperature = TEMP_MIN;

    (void)gba_create(CORE_PATH, ROM_PATH);
    startProcedure();

    while (1) {
        temperature = fmax(TEMP_MIN, TEMP_MAX * pow(TEMP_DECAY, (double)episode));

        trajectory** batch = malloc(WORKER_BATCH_SIZE * sizeof(trajectory*));
        assert(batch != NULL);
        for (int b=0; b<WORKER_BATCH_SIZE; b++) {
            batch[b] = runTrajectory(network, WORKER_STEPS, temperature);
        }

        for (int b=0; b<WORKER_BATCH_SIZE; b++) freeTrajectory(batch[b]);
        free(batch);

        if (stopCondition()) {
            gba_savestate(SAVESTATE_PATH);
            break;
        }
    }

    freeLSTM(network);
    gba_destroy();
    return 0;
    }